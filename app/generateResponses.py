from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from dotenv import load_dotenv, find_dotenv



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

def retrieve_inputs(file_path):
    # Read the paraquet dataset
    input_dataset = pd.read_parquet(file_path)
    return input_dataset


## Executor methods

def create_response_dataframe(index_name, input_dataset):
    """
    Create a RAG chain for the given index and vectorstore.
    """

    # Fetch vectorstore
    vectorstore = get_vectorstore(index_name)

    # Create the retriever (with Precision K=3 as requested)
    answers = []

    questions = input_dataset['input'].tolist()
    for q in questions:
        q_response = vectorstore.similarity_search(q, k=3)
        answers.append(q_response)

    output_dataframe = pd.DataFrame({'input': questions, 'output': answers})
    print(output_dataframe.head())

    return output_dataframe

def generate_response_pipeline(file_path, index_name):
    """
    Generate responses for the given input dataset.
    """

    # Retrieve input dataset
    input_dataset = retrieve_inputs(f'util/{index_name}_input_dataset.parquet')

    # Create response dataframe
    response_dataframe = create_response_dataframe(index_name, input_dataset)

    response_dataframe.to_csv(f'util/{index_name}_rag-test-conversations.csv', index=False)

    return response_dataframe


