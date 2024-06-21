from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from autorag.deploy import Runner
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch
import textwrap
import json
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
#from app.autorag_runner import Runner



"""
***RAG Chain for retrieving relevant content from a query***
Goals: To accurately retrieve relevant content to inform the response.
Status: IMPLEMENTED - V1.0
"""


# get API Keys
load_dotenv(find_dotenv())


# Helper methods
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_vectorstore(index_name):
    query_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Init the PineconeVectorStore for the index
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=query_embedding, namespace="MasterclassRetrieval")

    return vectorstore
def concatenate_content(conv_prefix):
    return "\n".join([entry['content'] for entry in conv_prefix])

def retrieve_inputs(file_path):
    # Read the paraquet dataset
    input_dataset = pd.read_parquet(file_path)

    # Extract and concatenate 'content' strings to a single input
    concatenated_content_list = [concatenate_content(row['conv_prefix']) for index, row in input_dataset.iterrows()]

    return concatenated_content_list




## Executor methods
def create_response_dataframe(index_name, input_list):
    """
    Create a RAG chain for the given index and vectorstore and perform a Pinecone similarity search
    """
    # Fetch vectorstore
    vectorstore = get_vectorstore(index_name)

    # Create the retriever (with Precision K=3 as requested)
    answers = []

    for q in input_list:
        q_response = vectorstore.similarity_search(q, k=3)
        answers.append(q_response)


    output_dataframe = pd.DataFrame({'input': input_list, 'output': answers})
    print(output_dataframe.head())

    return output_dataframe

def create_response_dataframe_autorag(input_list, yaml_dir, project_dir):
    """
    Create a RAG runner for the pipeline loaded and create responses for the given input list.
    """
    # Fetch AutoRAG Runner
    runner = Runner.from_yaml(yaml_dir, project_dir=project_dir)
    corpus_df = pd.read_parquet(f'{project_dir}/data/corpus.parquet')
    retrieved_metadata = []
    retrieved_contents = []

    for q in input_list:
        q_response_ids = runner.run(q, result_column="retrieved_ids")

        contents = corpus_df.loc[corpus_df['doc_id'].isin(q_response_ids), 'contents'].tolist()

        metadata = corpus_df.loc[corpus_df['doc_id'].isin(q_response_ids), 'metadata'].apply(
            lambda fm: {'int_id': fm['int_id'], 'kind': fm['kind'], 'title': fm['title']}).tolist()

        retrieved_contents.append(contents)
        retrieved_metadata.append(metadata)

    output_dataframe = pd.DataFrame({'input': input_list, 'contents': retrieved_contents, 'metadata': retrieved_metadata})
    output_dataframe.head()

    return output_dataframe


# Pipeline Methods

# Non-AutoRAG response generation pipeline
def generate_response_pipeline(index_name):
    """
    Generate responses for the given input dataset.
    """

    # Retrieve input dataset
    input_list = retrieve_inputs(f'util/{index_name}_input_dataset.parquet')

    # Create response dataframe
    response_dataframe = create_response_dataframe(index_name, input_list)
    response_dataframe.to_csv(f'util/{index_name}_rag-test-conversations.csv', index=False)

    return response_dataframe


# AutoRAG response generation pipeline
def generate_response_pipeline_autorag(index_name, yaml_dir='config/optimal.yaml', eval_version="",
                                       project_dir=f'{os.getcwd()}'):
    # Retrieve input dataset
    input_list = retrieve_inputs(f'util/{index_name}_input_dataset.parquet')
    response_df = create_response_dataframe_autorag(input_list, yaml_dir,project_dir)
    response_df.to_parquet(f'eval/{index_name}{eval_version}_rag-test-conversations.parquet', index=False)
    return response_df






