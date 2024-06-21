import os
import nest_asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset, Sequence, Value
import pandas as pd
from app.generateResponses import create_response_dataframe, generate_response_pipeline_autorag
from langchain_core.documents import Document
import logging
import uuid
from autorag.evaluator import Evaluator
from autorag.utils import cast_qa_dataset
import numpy as np
from autorag.deploy import extract_best_config
from app.documentProcessing import (init_multiprocessing,
                                    process_dataset_pipeline_parallel_autorag)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_synthetic_template(index_name, eval_version = "", use_existing=True):
    nest_asyncio.apply()
    if use_existing and os.path.isfile(f'util/{index_name}{eval_version}_template_eval.parquet'):
        print(f"Eval template set for {index_name}{eval_version} already exists. Skipping...")

        return Dataset.from_parquet(f'util/{index_name}{eval_version}_template_eval.parquet')
    else:
        documents = []
        df = pd.read_parquet(f"util/{index_name}{eval_version}_dataset.parquet")

        for index, row in df.iterrows():
            # Extract the content and metadata values
            content = row['content']
            metadata = {'int_id': row['int_id'], 'title': row['title'], 'kind': row['kind']}

            # Create a new Document object with the content and metadata
            document = Document(content, metadata=metadata)
            print(document.metadata)
            # Append the Document object to the list
            documents.append(document)


        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )
        testset = generator.generate_with_langchain_docs(documents, test_size=50,distributions={
                                                        simple: 0.5, reasoning: 0.4, multi_context: 0.10
                                                        })

        dataset = testset.to_dataset()
        dataset.to_csv(f"./util/{index_name}{eval_version}_template_eval.csv", index=False)
        return dataset


def eval_synthetic_dataset(index_name, dataset, eval_version = ""):

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
        ],
    )
    final_eval = result.to_pandas()
    final_eval.to_csv(f"util/{index_name}{eval_version}_final_eval.csv", index=False)
    return final_eval


def generate_context_response(index_name, eval_template_dataframe, eval_version = ""):

    # reformatting
    input_list = eval_template_dataframe['question'].tolist()

    output_dataframe = create_response_dataframe(index_name, input_list)

    # Process the 'output' data to extract the desired lists
    answers = output_dataframe["output"].tolist()
    answers_list = [[x.page_content for x in entry] for entry in answers]
    answer_metadata = [[x.metadata for x in entry] for entry in answers]

    # Rename columns as needed in the original dataframe
    eval_template_dataframe.rename(columns={'contexts': 'synth_contexts',
                                            'metadata': 'synth_metadata'}, inplace=True)

    # Bringing into Dataset format for Ragas
    final_dataset = Dataset.from_pandas(eval_template_dataframe)
    # Ragas tends to prefer additions inside the Dataset schema as PyArrow is a bit finicky
    try:
        final_dataset.remove_columns('__index_level_0__')
    except:
        logger.log(logging.INFO, "No erroneous columns to remove")
    final_dataset = final_dataset.add_column("contexts", answers_list)
    final_dataset = final_dataset.add_column("metadata", answer_metadata)

    final_dataset.to_csv(f"util/{index_name}_finaldata.csv")

    return final_dataset

def eval_responses_pipeline(index_name, eval_version = "", use_existing=True):
    logger.log(logging.INFO, "Generating synthetic eval template..")
    eval_template_dataset = build_synthetic_template(index_name, eval_version, use_existing)
    eval_template_dataframe = eval_template_dataset.to_pandas()

    logger.log(logging.INFO, "Generating Context Responses..")
    completed_eval_dataset = generate_context_response(index_name, eval_template_dataframe, eval_version)

    logger.log(logging.INFO, "Evaluating..")
    completed_eval_dataframe = eval_synthetic_dataset(index_name, completed_eval_dataset)

    logger.log(logging.INFO, "Completed evaluation")
    return completed_eval_dataframe



# re-wrote auto-rag package qa generation script to work directly with langchain documents (less overhead)
def generate_qa_autorag(distributions, generator_llm, critic_llm,
                        embedding_model, test_size, langchain_docs, **kwargs) -> pd.DataFrame:

    assert sum(list(distributions.values())) == 1.0, "Sum of distributions must be 1.0"

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embedding_model)

    test_df = generator.generate_with_langchain_docs(langchain_docs, test_size, distributions=distributions,
                                                     **kwargs).to_pandas()

    result_df = pd.DataFrame({
        'qid': [str(uuid.uuid4()) for _ in range(len(test_df))],
        'query': test_df['question'].tolist(),
        'generation_gt': list(map(lambda x: x, test_df['ground_truth'].tolist())),
    })
    print(result_df)
    result_df['retrieval_gt'] = test_df['metadata'].apply(lambda x: list(map(lambda y: y['doc_id'], x)))
    result_df = cast_qa_dataset(result_df)

    return result_df


def prepare_dataset_autorag(index_name, min_chunk_segments, eval_version):
    init_multiprocessing()
    results = process_dataset_pipeline_parallel_autorag(index_name, min_chunk_segments)
    rows = []
    print(results)
    for sublist in results:
        for doc in sublist:
            row = {'contents': doc.page_content,
                   'int_id': doc.metadata['int_id'],
                   'kind': doc.metadata['kind'],
                   'title': doc.metadata['title'],
                   'metadata': doc.metadata}
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.assign(doc_id=[str(uuid.uuid4()) for _ in range(len(df))])
    df.to_parquet(f'util/{index_name}{eval_version}_chunked{min_chunk_segments}_dataset_autorag.parquet')
    return df

def build_synthetic_template_autorag(index_name, eval_version="", use_existing=True, min_chunk_segments=3):
    print("building autorag template")
    if use_existing and os.path.isfile(f'util/{index_name}{eval_version}_template_eval_autorag.parquet'):
        print(f"Eval Autorag template set for {index_name}{eval_version} already exists. Skipping...")

        return pd.read_parquet(f'util/{index_name}{eval_version}_template_eval_autorag.parquet')
    else:

        print("building autorag template")
        # check if autorag configured dataset exists
        if not os.path.isfile(f'util/{index_name}{eval_version}_dataset_autorag.parquet'):
            prepare_dataset_autorag(index_name, min_chunk_segments, eval_version)

        df = pd.read_parquet(f"util/{index_name}{eval_version}_dataset_autorag.parquet")


        # use iloc to slice shorter to speed up process (and make my wallet not hurt as much)
        tenth_length = len(df) // 10
        first_tenth_df = df.iloc[:tenth_length]
        first_tenth_df.to_parquet(f"util/{index_name}{eval_version}_eval_dataset_autorag.parquet")

        documents = []
        for index, row in first_tenth_df.iterrows():
            # Extract the content and metadata values
            content = row['contents']
            metadata = {
                'doc_id': row['doc_id'], 'int_id': row['int_id'],
                'title': row['title'], 'kind': row['kind']
            }

            # Create a new Document object with the content and metadata
            document = Document(content, metadata=metadata)
            print(document.metadata)
            # Append the Document object to the list
            documents.append(document)

        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings()

        distributions = {  # uniform distribution
            simple: 0.25,
            reasoning: 0.45,
            multi_context: 0.30,
        }

        qa_df = generate_qa_autorag(distributions=distributions, generator_llm=generator_llm,
                                    critic_llm=critic_llm, embedding_model=embeddings, test_size=5,
                                    langchain_docs=documents)

        os.chdir("/Users/dcampbel/Nextcloud/Repositories/masterclassRetrieval")
        qa_df.to_parquet(f"./util/{index_name}{eval_version}_template_eval_autorag.parquet")
        return qa_df

def try_autorag(index_name, project_dir=os.getcwd(), eval_version="", qa_data_path = "", corpus_data_path = ""):

    if (qa_data_path == "" or corpus_data_path == ""):
        # ignoring if only one is set
        qa_data_path = f'util/{index_name}{eval_version}_template_eval_autorag.parquet'
        corpus_data_path = f'util/{index_name}_dataset_autorag.parquet'

    print(os.getcwd())
    evaluator = Evaluator(qa_data_path=qa_data_path,
                          corpus_data_path=corpus_data_path, project_dir=project_dir)
    print("Starting trial")
    evaluator.start_trial('full.yaml')

def build_child_to_master_synthetic_template_autorag(index_name, eval_version="", use_existing=True, min_chunk_segments=3):
    print("building child templates -> master template from core autorag dataset")
    df = pd.read_parquet('util/live_dataset_autorag.parquet')

    # identify unique id's
    unique_int_ids = df['int_id'].unique()

    # randomly select 5 unique int_ids for our child dfs (same document for each)
    selected_int_ids = np.random.choice(unique_int_ids, size=5, replace=False)

    # Create the list of child from all rows of the selected int_ids
    child_dataframes = [df[df['int_id'] == int_id] for int_id in selected_int_ids]

    print("building autorag templates for child dfs")
    qa_dfs = []
    for i, child_df in enumerate(child_dataframes):

        # now we generate the qa_df for each of the child dfs
        documents = []
        for index, row in child_df.iterrows():
            # Extract the content and metadata values
            content = row['contents']
            metadata = {
                'doc_id': row['doc_id'], 'int_id': row['int_id'],
                'title': row['title'], 'kind': row['kind']
            }

            # Create a new Document object with the content and metadata
            document = Document(content, metadata=metadata)
            print(document.metadata)
            # Append the Document object to the list
            documents.append(document)

        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings()

        distributions = {  # leaning towards multicontext
            multi_context: 1.0
        }

        qa_df = generate_qa_autorag(distributions=distributions, generator_llm=generator_llm,
                                    critic_llm=critic_llm, embedding_model=embeddings, test_size=11, # we want 10 but it often ends with 2 less
                                    langchain_docs=documents)

        os.chdir("/Users/dcampbel/Nextcloud/Repositories/masterclassRetrieval")
        qa_df.to_parquet(f"./util/{index_name}{eval_version}_child{i}_template_eval_autorag.parquet")
        qa_dfs.append(qa_df)

    # Merge the child dfs into one df for evaluation qa
    total_qa_df = pd.concat(qa_dfs, ignore_index=True)
    total_qa_df.to_parquet(f'util/{index_name}{eval_version}_child_qa_total_autorag.parquet')

    total_qa_path = f'util/{index_name}{eval_version}_child_qa_total_autorag.parquet'
    total_qa_df.head(5)
    return total_qa_path

def autorag_pipeline(index_name, project_dir, eval_version=""):
    build_synthetic_template_autorag(index_name, eval_version=eval_version, use_existing=True)
    print("Now entering autorag pipeline...")
    try_autorag(index_name=index_name, project_dir=project_dir, eval_version=eval_version)

def child_to_master_autorag_pipeline(index_name, eval_version="", use_original_corpus = True,
                                     project_dir=os.getcwd(), min_chunk_segments=3):
    if use_original_corpus:
        corpus_path = f'util/{index_name}_dataset_autorag.parquet'
    else:
        if not os.path.isfile(f'util/{index_name}{eval_version}_dataset_autorag.parquet'):
            prepare_dataset_autorag(index_name, min_chunk_segments, eval_version)
        corpus_path = f'util/{index_name}{eval_version}_dataset_autorag.parquet'

    total_qa_path = f'util/{index_name}{eval_version}_child_qa_total_autorag.parquet'
    #total_qa_path = f'util/{index_name}{eval_version}_template_eval_autorag_fquarter.parquet'
    if (os.path.isfile(total_qa_path) and
        os.path.isfile(corpus_path)):
        print("Existing files found, entering autorag evaluation")
        try_autorag(index_name="live", project_dir=project_dir, qa_data_path=total_qa_path, corpus_data_path=corpus_path, eval_version=eval_version)
    else:
        total_qa_path = build_child_to_master_synthetic_template_autorag(index_name, eval_version,
                                                    use_existing=True, min_chunk_segments=min_chunk_segments)

        print("Now entering autorag pipeline...")

        try_autorag(index_name="live",project_dir=project_dir,  qa_data_path=total_qa_path, corpus_data_path=corpus_path, eval_version=eval_version)


if __name__ == "__main__":
    os.chdir("..")
    # switch here for different pipelines
    runproject_name ="run3"
    trialnum_foryaml = "0"
    index_name = "live"
    eval_version = "v2"
    min_chunk_segments = 5
    yaml_path = f'{os.getcwd()}/config/{runproject_name}/optimal.yaml'
    project_dir = f'{os.getcwd()}/{runproject_name}'


    print("Starting autorag pipeline...")

    child_to_master_autorag_pipeline(index_name, project_dir=f'{os.getcwd()}/{runproject_name}',
                                     eval_version=eval_version, min_chunk_segments=min_chunk_segments)

    # autorag_pipeline("live", project_dir=f'{os.getcwd()}/{runproject_name}', eval_version="v1")


    # retrieve the best config
    print("retrieving best config...")
    if not os.path.isdir(f'{os.getcwd()}/config/{runproject_name}'):
        os.mkdir(f'{os.getcwd()}/config/{runproject_name}')

    extract_best_config(trial_path=f'{os.getcwd()}/{runproject_name}/{trialnum_foryaml}', output_path=yaml_path)

    # generate responses
    print("generating response df for inputs dataset")
    response_df = generate_response_pipeline_autorag(index_name, yaml_dir=yaml_path,
                                                     project_dir=project_dir, eval_version=eval_version)

    response_df.head()


