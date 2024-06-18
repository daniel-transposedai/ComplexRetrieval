import os
import nest_asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset, Sequence, Value
from langchain_community.document_loaders import CSVLoader
import pandas as pd
from app.generateResponses import create_response_dataframe
import logging
from langsmith import Client
from langsmith.utils import LangSmithError

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_synthetic_template(index_name, use_existing=True):
    nest_asyncio.apply()
    if use_existing and os.path.isfile(f'util/{index_name}_template_eval.json'):
        print(f"Eval template set for {index_name} already exists. Skipping...")

        return Dataset.from_json(f'util/{index_name}_template_eval.json')
    else:
        # Load the contents from the dataset
        loader = CSVLoader(f"util/{index_name}_dataset.csv", metadata_columns=['int_id', 'title', 'kind'])
        documents = loader.load()

        generator_llm = ChatOpenAI(model="gpt-4o")
        critic_llm = ChatOpenAI(model="gpt-4-turbo")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )
        testset = generator.generate_with_langchain_docs(documents, test_size=10,

                                                    distributions={simple: 0.5, reasoning: 0.4,
                                                                        multi_context: 0.10})
        dataset = testset.to_dataset()
        dataset.to_json(f"./util/{index_name}_template_eval.json", index=False)
        dataset.to_csv(f"./util/{index_name}_template_eval.csv", index=False)
        return dataset


def eval_synthetic_dataset(index_name, dataset):
    nest_asyncio.apply()

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
        ],
    )
    final_eval = result.to_pandas()
    final_eval.to_csv(f"util/{index_name}_final_eval.csv", index=False)
    return final_eval


def generate_context_response(index_name, eval_template_dataframe):

    # reformatting
    input_dataframe = pd.DataFrame({"input": eval_template_dataframe['question'].tolist()})

    output_dataframe = create_response_dataframe(index_name, input_dataframe)

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

def eval_responses_pipeline(index_name):
    logger.log(logging.INFO, "Generating synthetic eval template..")
    eval_template_dataset = build_synthetic_template(index_name)
    eval_template_dataframe = eval_template_dataset.to_pandas()

    logger.log(logging.INFO, "Generating Context Responses..")
    completed_eval_dataset = generate_context_response(index_name, eval_template_dataframe)

    logger.log(logging.INFO, "Evaluating..")
    completed_eval_dataframe = eval_synthetic_dataset(index_name, completed_eval_dataset)

    logger.log(logging.INFO, "Completed evaluation")
    return completed_eval_dataframe