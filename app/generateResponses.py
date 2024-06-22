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

def create_response_dataframe_autorag(input_path, yaml_dir, project_dir):
    """
    Create a RAG runner for the pipeline loaded and create responses for the given input list.
    """
    # Fetch AutoRAG Runner
    runner = Runner.from_yaml(yaml_dir, project_dir=project_dir)
    corpus_df = pd.read_parquet(f'{project_dir}/data/corpus.parquet')

    input_df = pd.read_parquet(input_path)
    input_list = input_df['conv_prefix'].tolist()

    retrieved_metadata = []
    retrieved_contents = []
    condensed_questions = []
    for q in input_list:
        print("context: " + str(q))
        condensed_q = condense_question(str(q).replace('{', '[').replace('}', ']'))
        condensed_questions.append(condensed_q)
        print("condensed context: " + condensed_q)
        q_response_ids = runner.run(condensed_q, result_column="retrieved_ids")

        contents = corpus_df.loc[corpus_df['doc_id'].isin(q_response_ids), 'contents'].tolist()

        metadata = corpus_df.loc[corpus_df['doc_id'].isin(q_response_ids), 'metadata'].apply(
            lambda fm: {'int_id': fm['int_id'], 'kind': fm['kind'], 'title': fm['title']}).tolist()

        retrieved_contents.append(contents)
        retrieved_metadata.append(metadata)

    content_columns = pd.DataFrame(retrieved_contents, columns=['context_0', 'context_1', 'context_2'])
    metadata_columns = pd.DataFrame(retrieved_metadata, columns=['metadata_0', 'metadata_1', 'metadata_2'])
    output_dataframe = pd.concat([input_df, content_columns, metadata_columns], axis=1)
    output_dataframe['condensed_question'] = condensed_questions

    output_dataframe.head()

    return output_dataframe

# Old - Basic vectorstore retrieval
def create_response_dataframe(index_name, input_list):
    """
    Create a RAG chain for the given index and vectorstore and perform a Pinecone similarity search
    """
    # Fetch vectorstore
    vectorstore = get_vectorstore(index_name)

    # Create the retriever (with Precision K=3 as requested)
    answers = []

    for q in input_list:
        q_response = vectorstore.similarity_search(str(q), k=3)
        answers.append(q_response)

    output_dataframe = pd.DataFrame({'input': input_list, 'output': answers})
    print(output_dataframe.head())

    return output_dataframe



# Message Chain Summarization
def condense_question(chat_history):
    condense_q_system_prompt = """Given a chat history between a user and an assistant \
        formulate the stand alone summary of information gathering statements and questions discussed by both parties so that they  \
        can be understood without the chat history. Do NOT answer the questions or statements, \
        just reformulate as needed and otherwise return it as is. \
        ONLY answer with the summarized output, and no other text. \
        Here is an example input chat history
        [('content': 'Hi Chris, great to talk to you!', 'metadata': None, 'role': 'user'), ('content': 'Hi there! Great to connect. What's on your mind?', 'metadata': None, 'role': 'assistant'), ('content': 'I was wondering if you could share a story about a negotiation that had a tight deadline.', 'metadata': None, 'role': 'user')] \
        In this input, the most relevant details are who is being discussed (Chris, in this example), and the request for a story about a negotiation with a tight deadline. The input may have multiple questions, and multiple relevant speakers, which is why you may need to create multiple standalone statements and questions."""


    print("condensing question")
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            ("human", "Chat History: {chat_history}"),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
            {"chat_history": RunnablePassthrough()}
            | condense_q_prompt
            | llm
            | StrOutputParser()
    )

    condensed_input = rag_chain.invoke(chat_history)


    return condensed_input


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
    input_path = f'util/{index_name}_input_dataset.parquet'
    response_df = create_response_dataframe_autorag(input_path, yaml_dir,project_dir)
    response_df.to_parquet(f'eval/{index_name}{eval_version}_rag-test-conversations.parquet', index=False)
    return response_df

if __name__ == "__main__":
    run = "runv8"
    yaml_dir = f'config/{run}/optimal.yaml'
    eval_ver = f"v7_minseg7_nofilterk3_broadcorpus"
    # Crucial that for outputs we utilize a pipeline that resolves at least k=3 outputs. No threshold limiting.
    response_df = generate_response_pipeline_autorag("live", yaml_dir=yaml_dir,
                                                     project_dir=run, eval_version=eval_ver)




