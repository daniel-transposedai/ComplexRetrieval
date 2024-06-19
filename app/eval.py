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
from app.generateResponses import create_response_dataframe
from langchain_core.documents import Document
import logging
import uuid
from autorag.evaluator import Evaluator
from autorag.utils import cast_qa_dataset
from autorag.data.corpus import llama_text_node_to_parquet
from app.documentProcessing import init_multiprocessing, process_dataset_pipeline_parallel_autorag

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_synthetic_template(index_name, eval_version = "", use_existing=True):
    nest_asyncio.apply()
    if use_existing and os.path.isfile(f'util/{index_name}{eval_version}_template_eval.csv'):
        print(f"Eval template set for {index_name}{eval_version} already exists. Skipping...")

        return Dataset.from_csv(f'util/{index_name}{eval_version}_template_eval.csv')
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

        # Old Method
        """print(os.getcwd())
        csv.field_size_limit(sys.maxsize)
        # Load the contents from the dataset
        pd.read_parquet(f'util/{index_name}{eval_version}_dataset.parquet').to_csv(
            f'{os.getcwd()}/util/{index_name}{eval_version}_dataset.csv')

        
        # create documents next
        loader = CSVLoader(f"util/{index_name}_dataset.csv", metadata_columns=['int_id', 'title', 'kind'])
        documents = loader.load()
        print(documents)
        """

        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )
        testset = generator.generate_with_langchain_docs(documents, test_size=50,

                                                    distributions={simple: 0.5, reasoning: 0.4,
                                                                        multi_context: 0.10})
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
    eval_template_dataframe.rename(columns={'contexts': 'synth_contexts', 'metadata': 'synth_metadata'}, inplace=True)

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
def generate_qa_autorag(distributions, generator_llm, critic_llm, embedding_model, test_size, langchain_docs, **kwargs) -> pd.DataFrame:
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


def prepare_dataset_autorag(index_name, eval_version=""):
    init_multiprocessing()
    results = process_dataset_pipeline_parallel_autorag(index_name)
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
    df.to_parquet(f'util/{index_name}{eval_version}_dataset_autorag.parquet')
    return df

def build_synthetic_template_autorag(index_name, eval_version="", use_existing=True):
    print("building autorag template")
    if use_existing and os.path.isfile(f'util/{index_name}{eval_version}_template_eval_autorag.parquet'):
        print(f"Eval Autorag template set for {index_name}{eval_version} already exists. Skipping...")

        return pd.read_parquet(f'util/{index_name}{eval_version}_template_eval_autorag.parquet')
    else:
        # check if autorag configured dataset exists
        if not os.path.isfile(f'util/{index_name}{eval_version}_dataset_autorag.parquet'):
            prepare_dataset_autorag(index_name, eval_version)

        print("building autorag template")

        df = pd.read_parquet(f"util/{index_name}{eval_version}_dataset_autorag.parquet")

        # use iloc to slice shorter
        quarter_length = len(df) // 10
        first_quarter_df = df.iloc[:quarter_length]

        documents = []
        for index, row in first_quarter_df.iterrows():
            # Extract the content and metadata values
            content = row['contents']
            metadata = {'doc_id': row['doc_id'], 'int_id': row['int_id'],'title': row['title'], 'kind': row['kind']}

            # Create a new Document object with the content and metadata
            document = Document(content, metadata=metadata)
            print(document.metadata)
            # Append the Document object to the list
            documents.append(document)

        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()

        distributions = {  # uniform distribution
            simple: 0.5,
            reasoning: 0.4,
            multi_context: 0.10,
        }

        qa_df = generate_qa_autorag(distributions=distributions, generator_llm=generator_llm, critic_llm=critic_llm,
                                    embedding_model=embeddings, test_size=50, langchain_docs=documents)

        os.chdir("/Users/dcampbel/Nextcloud/Repositories/masterclassRetrieval")
        qa_df.to_parquet(f"./util/{index_name}{eval_version}_template_eval_autorag.parquet")
        return qa_df


def try_autorag(index_name, eval_version=""):
    print(os.getcwd())
    evaluator = Evaluator(qa_data_path='util/live_template_eval_autorag.parquet', corpus_data_path='util/live_dataset_autorag.parquet')
    print("Starting trial")
    evaluator.start_trial('full.yaml')

if __name__ == "__main__":
    os.chdir("..")
    build_synthetic_template_autorag("live", use_existing=True)
    print("Now entering autorag pipeline...")
    try_autorag(index_name="live")


