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

def init_multiprocessing():
    if multiprocessing.get_start_method(allow_none=True) != 'forkserver':
        multiprocessing.set_start_method('forkserver', force=True)

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
    print("This ingestion is complete")
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
def applyHDPTopicModelSegmentation(texts, original_sentences, min_segment_size):
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Apply HDP and get dominant topics
    topic_boundaries = getDominantTopic_limited(corpus, dictionary, min_segment_size)

    # Segment the text based on topic boundaries
    segmented_texts = []
    start = 0
    for end in sorted(topic_boundaries):
        segmented_texts.append(' '.join(original_sentences[start:end]))
        start = end

    if start < len(original_sentences):
        segmented_texts.append(' '.join(original_sentences[start:]))

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

        dominant_topic = max(topic_probabilities, key=lambda x: x[1])[0] if topic_probabilities else None
        if current_topic != dominant_topic:
            topic_boundaries.add(i)
            current_topic = dominant_topic

    return topic_boundaries

def getDominantTopic_limited(corpus, dictionary, min_segment_size=3):

    # Applying the HDP model
    hdp = models.HdpModel(corpus, id2word=dictionary)

    # Topic Modelling for segmentation creates guide to split based on topic changes and retains context
    topic_boundaries = set()
    current_topic = None
    segment_size = 0
    for i, bow in enumerate(corpus):
        topic_probabilities = hdp[bow]

        dominant_topic = max(topic_probabilities, key=lambda x: x[1])[0] if topic_probabilities else None
        if current_topic != dominant_topic and segment_size >= min_segment_size:
            topic_boundaries.add(i)
            current_topic = dominant_topic
            segment_size = 0
        else:
            segment_size += 1

    return topic_boundaries


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



    print("Embedding and indexing documents")
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    return


# function to iterate over the dataset, apply topic modelling, and create documents
def process_dataset_with_HDP_serial(data_dict, index_name):

    # iterate over the dataset and create documents
    # we will want to do this in parallel. TODO: make this async
    for i, content in tqdm(enumerate(data_dict['content']), total=len(data_dict['content']), desc="Processing content"):
        # establish the metadata for this entry
        int_id = data_dict['int_id'][i]
        kind = data_dict['kind'][i]
        title = data_dict['title'][i]
        content_docs = []
        print("Preprocessing content texts with HDP")

        # preprocess the text of the content entry
        texts, original_sentences = preprocessText(content)
        # run HDP and topic segment the content text
        segmented_texts = applyHDPTopicModelSegmentation(texts, original_sentences)

        print("Formatting partitioned text into documents")
        # create the doc objects
        for text in segmented_texts:
            doc = Document(
                page_content = text,
                metadata = {"int_id": int_id, "kind": kind, "title": title}
            )
            content_docs.append(doc)

        # all_segments.append(text) Only do this if we want to embed across all entries
        print("Entering content docs embedding and Upsert to the VectorDB")
        # upsert and embed the docs to the VectorDB
        upsertToVectorDB(content_docs, index_name)
    print("Completed segmentation and Upsert of all content docs")
    return


def process_content_parallel(content, kind, title, int_id,  index_name):
    print("entered a single loop of this function")
    #multiprocessing.log_to_stderr("entered process content")
    try:
        content_docs = []
        print("Preprocessing content texts with HDP")

        # preprocess the text of the content entry

        texts, original_sentences = preprocessText(content)
        # run HDP and topic segment the content text
        segmented_texts = applyHDPTopicModelSegmentation(texts, original_sentences)

        print("Formatting partitioned text into documents")
        # create the doc objects
        for text in segmented_texts:
            doc = Document(
                page_content = text,
                metadata = {"int_id": int_id, "kind": kind, "title": title}
            )
            content_docs.append(doc)

        # all_segments.append(text) Only do this if we want to embed across all entries
        print("Entering content docs embedding and Upsert to the VectorDB")
        # upsert and embed the docs to the VectorDB
        upsertToVectorDB(content_docs, index_name)
        print("Upsert Complete")
        # Use a ProcessPoolExecutor to run the process_content function in parallel
    except Exception as e:
        print( f"Error processing content: {e}")
        raise e
    return content_docs




# Parallel intro step (no AutoRAG)
def process_dataset_with_HDP_parallel(data_dict, index_name):
    # Preparing data list for executor
    content_list = [(data_dict['content'][i], data_dict['kind'][i], data_dict['title'][i], data_dict['int_id'][i], index_name) for i in range(len(data_dict['content']))]

    results = []
    # Initialize tqdm progress bar

    # Using ProcessPoolExecutor to execute tasks in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each task individually
        future_to_content = {executor.submit(process_content_parallel_unpack, content): content for content in content_list}

        # As each task completes, update the progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_content), total=len(data_dict['content']), desc="Processing Entries"):
            result = future.result()  # Retrieve the result (or handle exceptions)
            results.append(result)

    print("Completed processing")
    return results





# Strategy 2 - Holistic embedding with HDP chunking
def process_content_parallel_autorag(content, kind, title, int_id, min_chunk_segments):
    print("entered a single loop of this function")
    #multiprocessing.log_to_stderr("entered process content")
    try:
        content_docs = []
        print("Preprocessing content texts with HDP")

        # preprocess the text of the content entry
        texts, original_sentences = preprocessText(content)
        # run HDP and topic segment the content text
        segmented_texts = applyHDPTopicModelSegmentation(texts, original_sentences, min_chunk_segments)

        print("Formatting partitioned text into documents")
        # create the doc objects
        for text in segmented_texts:
            doc = Document(
                page_content = text,
                metadata = {"int_id": int_id, "kind": kind, "title": title}
            )
            content_docs.append(doc)

    except Exception as e:
        print( f"Error processing content: {e}")
        raise e
    return content_docs


def process_content_parallel_unpack(args):
    return process_content_parallel(*args)

def process_content_parallel_unpack_autorag(args):
    return process_content_parallel_autorag(*args)

def process_dataset_with_HDP_parallel_autorag(data_dict, min_chunk_segments):

    # Preparing data list for executor
    content_list = [(data_dict['content'][i], data_dict['kind'][i], data_dict['title'][i], data_dict['int_id'][i], min_chunk_segments) for i in range(len(data_dict['content']))]

    results = []
    # Initialize tqdm progress bar

    # Using ProcessPoolExecutor to execute tasks in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each task individually
        future_to_content = {executor.submit(process_content_parallel_unpack_autorag, content): content for content in content_list}

        # As each task completes, update the progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_content), total=len(data_dict['content']), desc="Processing Entries"):
            result = future.result()  # Retrieve the result (or handle exceptions)
            results.append(result)

    print("Completed processing")
    return results



# End to End execution - Serial
def process_dataset_pipeline(index_name):
    print("Retrieving dataset")
    data_dict = ingest_dataset(f'util/{index_name}_dataset.parquet')
    print("Embedding and Upsert to the VectorDB")
    result_docs = process_dataset_with_HDP_serial(data_dict, index_name)
    return result_docs


# End to End execution - Parallel (for AutoRAG)
def process_dataset_pipeline_parallel_autorag(index_name, min_chunk_segments=3):
    print("Retrieving dataset")
    data_dict = ingest_dataset(f'util/{index_name}_dataset.parquet')
    result_docs = process_dataset_with_HDP_parallel_autorag(data_dict, min_chunk_segments)
    return result_docs


if __name__ == "__main__":
    init_multiprocessing()
    print(os.getcwd())
    os.chdir("/Users/dcampbel/Nextcloud/Repositories/masterclassRetrieval/")
    process_dataset_pipeline_parallel_autorag("live")