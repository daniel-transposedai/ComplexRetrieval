import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
import logging
import time
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
import multiprocessing


load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('punkt')

# Setup tokenizer and stopwords
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

model_name = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)


# function to ingress dataset into a pandas dataframe
def ingest_dataset(file_path):
    # Read the paraquet dataset
    content_dataset = pd.read_parquet(file_path)
    # Might be more here to do, we may want to split this out

    contents = content_dataset['content'].tolist()
    int_ids = content_dataset['int_id'].tolist()
    titles = content_dataset['title'].tolist()
    kinds = content_dataset['kind'].tolist()
    data_dict = {
        'content': contents,
        'int_id': int_ids,
        'title': titles,
        'kind': kinds
    }
    return data_dict

# Function to read and preprocess dataset csv from a file
def preprocessText(text):
    documents = sent_tokenize(text)
    texts = [
        [word for word in tokenizer.tokenize(document) if word not in stop_words]
        for document in documents
    ]
    return texts, documents

# Function to apply Hierarchical Dirichlet Process (HDP) Topic Modeling
def applyHDPTopicModelSegmentation(texts, original_sentences):
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Apply HDP and get dominant topics
    topic_boundaries = getDominantTopic(corpus, dictionary)

    # Segment the text based on topic boundaries
    segmented_texts = []
    start = 0
    for end in sorted(topic_boundaries):
        segmented_texts.append(' '.join(original_sentences[start:end]))
        start = end

    if start < len(original_sentences):
        segmented_texts.append(' '.join(original_sentences[start:]))

    # Example of segments
    for segment in segmented_texts:
        print(segment)

    # Remove any empty string stragglers
    segmented_texts = [text for text in segmented_texts if text.strip()]

    return segmented_texts

def getDominantTopic(corpus, dictionary):

    # Applying the HDP model
    hdp = models.HdpModel(corpus, id2word=dictionary)

    # Topic Modelling for segmentation creates guide to split based on topic changes and retains context
    topic_boundaries = set()
    current_topic = None
    for i, bow in enumerate(corpus):
        topic_probabilities = hdp[bow]
        logger.log(logging.INFO, "Retrieved topic probabilities " + str(topic_probabilities))
        dominant_topic = max(topic_probabilities, key=lambda x: x[1])[0] if topic_probabilities else None
        if current_topic != dominant_topic:
            topic_boundaries.add(i)
            current_topic = dominant_topic
    logger.log(logging.INFO, "Completed HDP with topic boundaries " + str(topic_boundaries))

    return topic_boundaries

# Function to get embeddings using OpenAI API
"""def getEmbeddings(texts):
    client = OpenAI()
    embeddings = []
    for text in texts:
        response = client.embeddings.create(input=text,
        model="text-embedding-3-large")
        embeds = [record.embedding for record in response.data]
        embeddings += embeds
    return embeddings"""

def upsertToVectorDB(documents, index_name):

    # Initialize Pinecone
    print("Entering Upsert Method")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = index_name
    namespace = "MasterclassRetrieval"

    # Create or connect to an existing index
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=3072,  # dimensionality of text-embed-3-large
            metric='dotproduct',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)



    logger.log(logging.INFO, "Embedding and indexing documents")
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    return


# function to iterate over the dataset, apply topic modelling, and create documents
def process_dataset_with_HDP(data_dict, index_name):

    #all_segments = [] Only do this if we want to embed across all entries

    # iterate over the dataset and create documents
    # we will want to do this in parallel. TODO: make this async
    for i, content in tqdm(enumerate(data_dict['content']), total=len(data_dict['content']), desc="Processing content"):
        # establish the metadata for this entry
        int_id = data_dict['int_id'][i]
        kind = data_dict['kind'][i]
        title = data_dict['title'][i]
        content_docs = []
        logger.log(logging.INFO, "Preprocessing content texts with HDP")

        # preprocess the text of the content entry
        texts, original_sentences = preprocessText(content)
        # run HDP and topic segment the content text
        segmented_texts = applyHDPTopicModelSegmentation(texts, original_sentences)

        logger.log(logging.INFO, "Formatting partitioned text into documents")
        # create the doc objects
        for text in segmented_texts:
            doc = Document(
                page_content = text,
                metadata = {"int_id": int_id, "kind": kind, "title": title}
            )
            content_docs.append(doc)
        logger.log(logging.INFO, "Created docs " + str(content_docs))

        # all_segments.append(text) Only do this if we want to embed across all entries
        logger.log(logging.INFO, "Entering content docs embedding and Upsert to the VectorDB")
        # upsert and embed the docs to the VectorDB
        upsertToVectorDB(content_docs, index_name)
    logger.log(logging.INFO, "Completed segmentation and Upsert of all content docs")
    return

def process_dataset_with_HDP_parallel(data_dict, index_name):
    def process_content(index_content_tuple):
        multiprocessing.log_to_stderr("entered process content")
        try:
            i, content = index_content_tuple
            print(content)
            # establish the metadata for this entry
            int_id = data_dict['int_id'][i]
            kind = data_dict['kind'][i]
            title = data_dict['title'][i]
            content_docs = []
            logger.log(logging.INFO, "Preprocessing content texts with HDP")

            # preprocess the text of the content entry
            multiprocessing.log_to_stderr("preparing to enter preprocess")

            texts, original_sentences = preprocessText(content)
            # run HDP and topic segment the content text
            segmented_texts = applyHDPTopicModelSegmentation(texts, original_sentences)

            logger.log(logging.INFO, "Formatting partitioned text into documents")
            # create the doc objects
            for text in segmented_texts:
                doc = Document(
                    page_content = text,
                    metadata = {"int_id": int_id, "kind": kind, "title": title}
                )
                content_docs.append(doc)
            logger.log(logging.INFO, "Created docs " + str(content_docs))

            # all_segments.append(text) Only do this if we want to embed across all entries
            logger.log(logging.INFO, "Entering content docs embedding and Upsert to the VectorDB")
            # upsert and embed the docs to the VectorDB
            upsertToVectorDB(content_docs, index_name)
            # Use a ProcessPoolExecutor to run the process_content function in parallel
        except Exception as e:
            logger.log(logging.ERROR, f"Error processing content: {e}")
            return

    with concurrent.futures.ProcessPoolExecutor() as executor:
        logger.info("About to call executor.map")
        executor.map(process_content, enumerate(data_dict['content']))
        logger.info("Finished calling executor.map")

    logger.log(logging.INFO, "Completed segmentation and Upsert of all content docs")
    return

# TODO: function to iterate over the dataset, apply topic modelling, and create documents
def process_dataset_without_chunking(data_dict):
    return None

# End to End execution
def process_dataset_pipeline(file_path, index_name):
    logger.log(logging.INFO, "Retrieving dataset")
    data_dict = ingest_dataset(file_path)
    logger.log(logging.INFO, "Embedding and Upsert to the VectorDB")
    process_dataset_with_HDP(data_dict, index_name)
    return
