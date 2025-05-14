import argparse
import glob
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

from llama_index.core import ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.prompts.base import PromptTemplate, BasePromptTemplate
from llama_index.core.schema import BaseNode, Document, TextNode

from utils import (
    read_markdown_files,
    create_service_context
) 

from langchain_core.language_models import BaseLanguageModel

MAX_TOKENS = 10000 #16384
MAX_NODES = 10

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))  # Redirecting to stdout to see logs in Azure
HOME = '/home/ksartik/farmvibes-llm/agkgcopilot'

ONTOLOGY_JSONLD_DATA_CREATE_TMPL = """
Here is a context definition for an ontology.

Context Definition:

{context_definition}

-----------------

Generate a JSON-LD using the following data and the above context definition for the given ontology.
Use '@graph' object namespace for the data in JSON-LD.
Be comprehensive and make sure to fill all of the data completely WITHOUT leaving the sentence in "...".
If there are multiple subfields enumerated in a 'List' namespace then do not combine them in a single subfield, keep them as separate subfields to disambiguate.
Ensure that you populate all items in the 'List' namespace, do not leave any item.
Do not include any explanations or apologies in your response.
Do not add any other text other than the generated JSON-LD in your response
Generate in Json format. 
----------------------
Data:

{data}
---------------------
JSON-LD json:
"""  # noqa: E501

# Keep nesting to the minimum and still be able to disambiguate.

ONTOLOGY_JSONLD_DATA_CREATE_PROMPT = PromptTemplate(ONTOLOGY_JSONLD_DATA_CREATE_TMPL, prompt_type="structured_fill")


def map_ontology(
    ontology_file_path: str,
    service_context: ServiceContext,
    input_dir: str,
    output_dir: str,
) -> None:

    subdirectories = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    if not subdirectories:
        subdirectories = [Path(input_dir)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for subdir in subdirectories:
            subdir_name = subdir.stem
            LOGGER.info(f"Processing subdirectory: {subdir}")

            try:
                markdown_paths = list(subdir.glob("*.md"))
                LOGGER.info(f"Checking all markdowns in this subdirectory: {markdown_paths}")

                documents = read_markdown_files(markdown_paths)

                if not documents:
                    LOGGER.info(f"No markdown files found in {subdir_name}, skipping.")
                    continue

                if not Path(ontology_file_path).is_file():
                    raise FileNotFoundError(f"No JSON file found at the specified path: {ontology_file_path}")

                current_output_dir = Path(output_dir) / f"{subdir_name}_ontology"
                current_output_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Creating new output subdirectory: {current_output_dir}")

                ontology_creator = OntologyMapping(
                    ontology_context_definition_path=ontology_file_path,
                    service_context=service_context,
                    documents=documents,
                    ontology_jsonld_data_create_prompt=ONTOLOGY_JSONLD_DATA_CREATE_PROMPT,
                )

                ontology_creator.generate_and_save_ontology_data(executor, current_output_dir)  # type: ignore

                LOGGER.info(f"Finished processing subdirectory: {subdir_name}")
            except Exception as e:
                LOGGER.error(f"An error occurred while processing subdirectory {subdir.stem}: {e}")
                continue

class OntologyMapping_old:
    def __init__(
        self,
        ontology_context_definition_path: str,
        documents: Sequence[Document], #Optional[Sequence[Document]] = None,
        service_context: Optional[ServiceContext] = None,
        ontology_jsonld_data_create_prompt: Optional[BasePromptTemplate] = ONTOLOGY_JSONLD_DATA_CREATE_PROMPT,
        _fmt_nodes_g2create_prompt: Optional[List[str]] = None,
        chunk_size: int=8192
    ) -> None:
        self.service_context = service_context
        self.documents = documents
        self._nodes = self.get_nodes(chunk_size=chunk_size)
        self.ontology_context_definition = self.load_context_definition(ontology_context_definition_path)
        self._fmt_nodes_g2create_prompt = _fmt_nodes_g2create_prompt or []  # Use provided list or initialize as empty
        self.ontology_jsonld_data_create_prompt = ontology_jsonld_data_create_prompt

    def get_nodes(self, chunk_size: int) -> List[BaseNode]:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        return node_parser.get_nodes_from_documents(self.documents)

    def load_context_definition(self, json_file_path: str) -> str:
        with open(json_file_path) as f:
            return f.read()

    def save_ontology_to_json(self, ontology_data: str, filename: str) -> None:
        with open(filename, "w") as json_file:
            json_file.write(ontology_data)

    def process_and_save_node(self, node: BaseNode, idx: int, output_dir: str) -> None:
    # def process_and_save_node(self, node: TextNode, idx: int, output_dir: str) -> None:
        LOGGER.info(f"Processing Node {idx}. Generating data for ontology.")
        data = node.text if isinstance(node, TextNode) else None
        prompt = self.ontology_jsonld_data_create_prompt.format(  # type: ignore
            data=data, context_definition=self.ontology_context_definition
        )
        self._fmt_nodes_g2create_prompt.append(prompt)  # Store the prompt
        # LOGGER.info(f"Complete prompt: {self._fmt_nodes_g2create_prompt}")
        jsonld_data = self.service_context.llm.complete(prompt).text  # type: ignore
        json_filename = os.path.join(output_dir, f"ontology_node_{idx}.jsonld")
        self.save_ontology_to_json(jsonld_data, json_filename)
        LOGGER.info(f"Saved ontology data from node {idx} to {json_filename}")

    def generate_and_save_ontology_data(self, executor: ThreadPoolExecutor, output_dir: str) -> None:
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            futures = [
                executor.submit(self.process_and_save_node, node, idx, output_dir)
                for idx, node in enumerate(self._nodes)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"An error occurred while processing a node: {e}")
        except Exception as e:
            LOGGER.error(f"An error occurred during the ontology data generation process: {e}")
    


class OntologyMapping:
    def __init__(
        self,
        ontology_context_definition_path: str,
        documents: Sequence[Document], #Optional[Sequence[Document]] = None,
        llm: BaseLanguageModel,
        ontology_jsonld_data_create_prompt: Optional[BasePromptTemplate] = ONTOLOGY_JSONLD_DATA_CREATE_PROMPT,
        _fmt_nodes_g2create_prompt: Optional[List[str]] = None,
        chunk_size: int=8192
    ) -> None:
        self.llm = llm
        self.documents = documents
        self._nodes = self.get_nodes(chunk_size=chunk_size)
        self.ontology_context_definition = self.load_context_definition(ontology_context_definition_path)
        self._fmt_nodes_g2create_prompt = _fmt_nodes_g2create_prompt or []  # Use provided list or initialize as empty
        self.ontology_jsonld_data_create_prompt = ontology_jsonld_data_create_prompt

    def get_nodes(self, chunk_size: int) -> List[BaseNode]:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        return node_parser.get_nodes_from_documents(self.documents)

    def load_context_definition(self, json_file_path: str) -> str:
        with open(json_file_path) as f:
            return f.read()

    def save_ontology_to_json(self, ontology_data: str, filename: str, text: str=None) -> None:
        with open(filename, "w") as json_file:
            json_file.write(ontology_data)
        if text:
            import json
            with open(filename, 'r') as f:
                data = json.load(open(filename, "r"))
                data['@text'] = text
            with open(filename, 'w') as wf:
                json.dump(data, wf)

    def process_and_save_node(self, node: BaseNode, idx: int, output_dir: str) -> None:
    # def process_and_save_node(self, node: TextNode, idx: int, output_dir: str) -> None:
        LOGGER.info(f"Processing Node {idx}. Generating data for ontology.")
        data = node.text if isinstance(node, TextNode) else None
        prompt = self.ontology_jsonld_data_create_prompt.format(  # type: ignore
            data=data, context_definition=self.ontology_context_definition
        )
        self._fmt_nodes_g2create_prompt.append(prompt)  # Store the prompt
        # LOGGER.info(f"Complete prompt: {self._fmt_nodes_g2create_prompt}")
        jsonld_data = self.llm.invoke(prompt, max_tokens=MAX_TOKENS, 
                                      response_format={"type": "json_object"}).content  # type: ignore
        json_filename = os.path.join(output_dir, f"ontology_node_{idx}.jsonld")
        self.save_ontology_to_json(jsonld_data, json_filename) #, text=data)
        LOGGER.info(f"Saved ontology data from node {idx} to {json_filename}")

    def generate_and_save_ontology_data(self, executor: ThreadPoolExecutor, output_dir: str) -> None:
        for start_idx in range(0, len(self._nodes), MAX_NODES):
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                futures = [
                    executor.submit(self.process_and_save_node, node, start_idx+idx, output_dir)
                    for idx, node in enumerate(self._nodes[start_idx:min(start_idx+MAX_NODES, len(self._nodes))])
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        LOGGER.error(f"An error occurred while processing a node: {e}")
            except Exception as e:
                LOGGER.error(f"An error occurred during the ontology data generation process: {e}")
    

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Process ontology nodes and save to JSON files.")
        parser.add_argument(
            "-p",
            "--ontology_path",
            required=True,
            # default=f'{HOME}/data/ontology/',
            help="Path to the ontology context definition JSON-LD file.",
        )
        parser.add_argument(
            "-i",
            "--input_dir",
            # required=True,
            default=f'{HOME}/data/md/',
            help="Input directory containing the markdown files to be processed.",
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            default=f'{HOME}/data/md/',
            # required=True,
            help="Directory where the JSON-LD files will be saved. " "Defaults to the current directory.",
        )
        parser.add_argument("-d", "--deployment_name", help="The deployment name for OpenAI.")
        parser.add_argument(
            "-e",
            "--embedding_model",
            default="sentence-transformers/all-mpnet-base-v2",
            help="The embedding model to use.",
        )
        parser.add_argument(
            "--connection_id",
            type=str,
            # required=True,
            default=None,
            help="AML connection id for OpenAI",
        )

        args = parser.parse_args()
        embedding_model = args.embedding_model

        service_context = create_service_context(args.connection_id, embedding_model, args.deployment_name)

        map_ontology(
            ontology_file_path=args.ontology_path,
            service_context=service_context,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

    except Exception as e:
        LOGGER.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
