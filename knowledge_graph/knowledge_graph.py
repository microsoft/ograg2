"""
This module provides functionality for working with a knowledge graph.
"""

import argparse
import ast
import glob
import json
import logging
import os
import pickle
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from retrying import retry

from llama_index.core import ServiceContext
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

from utils import create_service_context
from langchain_core.language_models import BaseLanguageModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

DEFAULT_KG_TRIPLET_ONTOLOGY_EXTRACT_TMPL = """
Using the @graph namespace in the following json-ld, generate a complete python list of tuples of triples for knowledge graph in the format (subject, predicate, object).
Keep the property names exactly as it is in the Json-ld.
The 'subject', 'predicate', and 'object' can only be strings.
Subjects and objects should be in natural language.
Make sure that the predicate is structured so that it is a grammatically correct phrase.
The triples cannot be nested, so please flatten them. Also do not include triples keys of "subject", "object", "predicate", only the values.
For nested structure, within "@graph" object, such as "xy": {{"k": "v"}} flatten it by rearranging keys "xy", "k" to either "xyk", "xky", or "kxy" in a way that it grammatically makes sense.
Generate all triples.
Do not add any other text in response other than the list of tuples of triples.
------------------------------

JSON-LD:
{data}

"""  # noqa: E501 W605

KG_TRIPLET_ONTOLOGY_EXTRACT_TMPL = """
Using the @graph namespace in the following json-ld, generate a complete python list of tuples of triples for knowledge graph in the format (subject, predicate, object).
Keep the property names exactly as it is in the Json-ld, which is provided in the 'name' key for complex fields and directly as values for lists or strings.
The 'subject', 'predicate', and 'object' can only be strings.
The triples cannot be nested, so please flatten them. Also do not include triples keys of "subject", "object", "predicate", only the values.
While constructing the predicate during flattening of nested fields, include the names of all the parent subject nodes in predicate.
For example, an ontology snippet of nested fields and the generated Triplets are provided below:
--------------------------------------
Example of Ontology snippet:
"{{"@graph": [
        {{
            "@type": "Crop",
            "name": "Wheat",
			"has_types": [
                {{
                    "@type": "CropType",
                    "name": "Triticum aestivum",
                    "used_for": "chapati and bakery products"
                }}
			]
            "has_growing_zones": {{
                "@type": "cropCult:CropGrowingZones",
                "CropGrowingZone": [
                    {{
                        "name": "North Western Plains Zone",
                        "has_seed_recommendations": {{
                            "@type": "cropCult:SeedList",
                            "variety_name": ["KRL 19", "PBW 502"],
                            "has_early_sowing_time": {{
                                "@type": "cropCult:SowingTime",
                                "start_date": "1st November",
                                "end_date": "20th November"
                            }}
                        }}
                    }}
				]
			}}
		}}
    ]
}}"

Generated Triplets:
"[('Wheat', 'has type', 'Triticum aestivum'),
('North Western Plains Zone', 'Wheat has seed recommendation variety', 'KRL 19'),
('North Western Plains Zone', 'Wheat has seed recommendation variety', 'PBW 502'),
('KRL 19', 'Wheat North Western Plains Zone has early sowing time start date', '1st November'),
('KRL 19', 'Wheat North Western Plains Zone has early sowing time end date', '20th November'),
('PBW 502', 'Wheat North Western Plains Zone has early sowing time start date', '1st November'),
('PBW 502', 'Wheat North Western Plains Zone has early sowing time end date', '20th November')]"

In this example, "Wheat" is the name of the parent subject of "North Western Plains Zone", while "North Western Plains Zone" is the parent subject of "KRL 19" and "PB 502".
Therefore, both these subjects are mentioned in the appropriate triplet predicates.
Note how these subjects are enumerated within the predicate, starting with the first order field, then second and so on, until it reaches the leaf node.
--------------------------
Subjects and objects should be in natural language.
Generate all triples.
Do not add any other text in response other than the list of tuples of triples.
------------------------------

JSON-LD:
{data}
"""  # noqa: E501 E101 W191

KG_TRIPLET_ONTOLOGY_EXTRACT_PROMPT = PromptTemplate(
    KG_TRIPLET_ONTOLOGY_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)


def retry_if_ast_eval_error(exception: Exception) -> bool:
    """Return True if we should retry (in this case when it's an SyntaxError), False otherwise"""
    return isinstance(exception, SyntaxError)


def safe_literal_eval(triples_str: str) -> List[Tuple[str, str, str]]:
    """Safely evaluate the string representation of triples."""
    return ast.literal_eval(triples_str)


def create_kg_triples(
    input_directory: str,
    output_directory: str,
    llm: BaseLanguageModel,
    batch_size: int,
) -> None:
    """
    Parse command-line arguments and runs the knowledge graph generation process.
    """
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:

        for subdir_name in os.listdir(input_directory):
            LOGGER.info(f"Started processing: {subdir_name}")
            subdir_path = os.path.join(input_directory, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            json_filenames = glob.glob(os.path.join(subdir_path, "*.jsonld"))
            all_triples_with_documents = []  # This will store tuples of (document_name, subject, predicate, object)
            triple_generator = KGGenerator(llm=llm)

            output_pickle_path = os.path.join(output_directory, f"{subdir_name}_triples.pkl")

            futures = [
                executor.submit(triple_generator.generate_triples, json_filename) for json_filename in json_filenames
            ]

            for future in as_completed(futures):
                try:
                    triples_with_document = future.result()
                    all_triples_with_documents.extend(triples_with_document)
                    json_filename = json_filenames[
                        futures.index(future)
                    ]  # Get the filename associated with the future
                    LOGGER.info(f"Finished processing file {json_filename}")
                except Exception as e:
                    LOGGER.error(f"An error occurred while processing {json_filename}: {e}")
                finally:
                    if len(all_triples_with_documents) >= batch_size:
                        triple_generator.append_triples_to_pkl(all_triples_with_documents, output_pickle_path)
                        all_triples_with_documents = []  # Reset the list after saving

        if all_triples_with_documents:
            triple_generator.append_triples_to_pkl(all_triples_with_documents, output_pickle_path)

        LOGGER.info(f"All triples with document names for {subdir_name} " f"have been saved to {output_pickle_path}")


def process_file(json_filename: str) -> List[Tuple[str, str, str]]:
    """
    Process the files in the input directory and saves the results to the output directory.
    """
    with open(json_filename, "r") as json_file:
        ontology_data = json.load(json_file)

    ontology_data_str = json.dumps(ontology_data)
    triple_generator = KGGenerator(service_context=service_context)
    triples = triple_generator.generate_triples(ontology_data_str)
    LOGGER.info(f"Processed triples from {json_filename}.")
    return triples  # type: ignore

class KGGenerator:
    """
    Generate a knowledge graph from a set of triples.
    """

    def __init__(self, llm: BaseLanguageModel, verbose: bool = True):
        self.llm = llm
        self._verbose = verbose

    @retry(stop_max_attempt_number=3, retry_on_exception=retry_if_ast_eval_error)
    def generate_triples(self, json_filename: str) -> List[Tuple[str, str, str, str]]:
        """Generate and evaluate triples from ontology data file."""
        with open(json_filename, "r") as json_file:
            ontology_data_str = json_file.read()
        prompt = KG_TRIPLET_ONTOLOGY_EXTRACT_PROMPT.format(data=ontology_data_str)
        triples_str = self.llm.invoke(prompt).content
        triples = ast.literal_eval(triples_str)
        document_name = os.path.basename(json_filename).replace(".jsonld", "")
        triples_with_document = [(document_name,) + triple for triple in triples]
        return triples_with_document

    def save_triples_to_pkl(self, triples: List[Tuple[str, str, str]], filename: str):
        """Save triples to a pickle file."""
        with open(filename, "wb") as pkl_file:
            pickle.dump(triples, pkl_file)

    def append_triples_to_pkl(self, triples: List[Tuple[str, str, str]], filename: str):
        """Append triples to a pickle file."""
        with open(filename, "ab+") as pkl_file:
            pickle.dump(triples, pkl_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process triples from Knowledge Graph from JSON-LD files.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        # default='data/ontology/',
        help="Input path to the directories containing the JSON-LD files.",
    )
    parser.add_argument(
        "-o", 
        "--output", 
        # required=True,
        default='data/kg/pkl',
        help="Output path to save the PKL files.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Number of triples to process before saving to disk.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="The embedding model to use.",
    )
    parser.add_argument(
        "-ci",
        "--connection_id",
        type=str,
        default=None,
        # required=True,
        help="AML connection id for OpenAI",
    )
    parser.add_argument("-d", "--deployment_name", help="The deployment name for OpenAI.")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    service_context = create_service_context(args.connection_id, args.embedding_model, args.deployment_name)

    create_kg_triples(
        input_directory=args.input,
        output_directory=args.output,
        service_context=service_context,
        batch_size=args.batch_size,
    )
