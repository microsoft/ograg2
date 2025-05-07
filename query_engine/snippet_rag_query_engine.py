from llama_index.core.prompts import PromptTemplate, PromptType
from typing import Any, DefaultDict, Dict, List, Optional
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode
import json
import os
from llama_index.core import VectorStoreIndex
from tqdm import tqdm

QUERY_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Answer: """


CONVERT_PROMPT = """Convert the following information about an entity into an english sentence. 
The information is presented as a list of {{key: value}} where key is a property name and the value is its value.
Remove any redundant information but KEEP ALL the information that is important. DO NOT COMPRESS INFORMATION USING "and so on" or "etc" or "and others" etc.
---------------------
For example, 
Information: {{'name': 'John Doe', 'age': '25', 'location': 'New York'}}
Sentence: John Doe is 25 years old and lives in New York.

Information: {{'@type': 'Crop', 'name': 'Soybean', 'seed_germination_test_requirements_are': 'Seed Germination Test Requirements'}}
Sentence: The crop soybean has Seed Germination Test Requirements.

---------------------

Information: {information}
---------------------
Sentence: 
"""


def merge_dicts(dict1, dict2):
    new_dict = {}
    for key, value in dict1.items():
        if (key in dict2) and (dict1[key] != dict2[key]):
            return None
        else:
            new_dict[key] = value
    for key, value in dict2.items():
        new_dict[key] = value
    return new_dict

def flatten_tree (node):
    flattened_nodes = []
    node_context = {k: v for k, v in node.items() if not isinstance(v, dict) and not isinstance(v, list)}
    flattened_nodes.append(node_context)
    for k, v in node.items():
        if isinstance(v, dict):
            for v_node in flatten_tree(v):
                v_node = {f'{k}_{k2}': v2 for k2, v2 in v_node.items()}
                flattened_nodes.append({**node_context, **v_node})
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for v_node in flatten_tree(item):
                        v_node = {f'{k}_{k2}': v2 for k2, v2 in v_node.items()}
                        flattened_nodes.append({**node_context, **v_node})
                else:
                    flattened_nodes.append({**node_context, k: item})
    return flattened_nodes

def load_graph_nodes(ontology_nodes_path: str):
        nodes = []
        for ontdir in os.listdir(ontology_nodes_path):
            ontdir = os.path.join(ontology_nodes_path, ontdir)
            if os.path.isdir(ontdir) and ontdir.endswith('ontology'):
                for fname in os.listdir(ontdir):
                    if fname.endswith('.jsonld'):
                        fname = os.path.join(ontdir, fname)
                        with open(f'{fname}', 'r') as f:
                            nodes += json.load(f)['@graph']
        return nodes

class SnippetRAGQueryEngine:
    def __init__(self, llm: BaseLanguageModel, vector_retriever: VectorIndexRetriever):
        self._llm = llm
        self._vector_retriever = vector_retriever


    @classmethod
    def from_vector_retriever(cls, llm: BaseLanguageModel, vector_retriever: VectorIndexRetriever):
        return cls(
            llm=llm,
            vector_retriever=vector_retriever
        )
    
    @classmethod
    def from_vector_base(cls, llm: BaseLanguageModel, vector_retriever: VectorStoreIndex):
        return cls(
            llm=llm,
            vector_retriever=VectorIndexRetriever(index=vector_retriever)
        )

    @classmethod
    def from_ontology_path(
        cls, 
        ontology_nodes_path: str, 
        llm: BaseLanguageModel, 
        embed_model: Embeddings,
        show_progress: bool = True
    ):
        if 'vector_store' in os.listdir(ontology_nodes_path):
            from llama_index.core.storage.storage_context import StorageContext
            from llama_index.core import load_index_from_storage
            vector_storage_context = StorageContext.from_defaults(persist_dir=f"{ontology_nodes_path}/vector_store")
            index = load_index_from_storage(vector_storage_context)
            vector_retriever = VectorIndexRetriever(index=vector_index)
            return cls(
                llm=llm,
                vector_retriever=vector_retriever
            )
        else:
            if 'flattened_texts.txt' in os.listdir(ontology_nodes_path):
                with open(f'{ontology_nodes_path}/flattened_texts.txt', 'r') as f:
                    texts = f.read().split('\n')
            else:
                nodes = []
                for node in load_graph_nodes(ontology_nodes_path):
                    nodes += flatten_tree(node)

                new_nodes = []
                for node in nodes:
                    for node2 in nodes:
                        if node != node2:
                            merged = merge_dicts(node, node2)
                            if merged:
                                new_nodes.append(merged)

                texts = []
                for node in tqdm(nodes, desc='Converting nodes to text') if show_progress else nodes:
                    texts.append(llm.invoke(CONVERT_PROMPT.format(information=node), max_tokens=MAX_TOKENS).content)
                with open(f'{ontology_nodes_path}/flattened_texts.txt', 'w') as f:
                    f.write('\n'.join(texts))
            embs = embed_model.embed_documents(texts)
            nodes = [TextNode(text=text, embedding=emb) for text, emb in zip(texts, embs)]
            vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
            vector_index.storage_context.persist(f"{ontology_nodes_path}/vector_store")
        vector_retriever = VectorIndexRetriever(index=vector_index)
        return cls(
            llm=llm,
            vector_retriever=vector_retriever
        )

    def _retrieve_nodes(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        return retrieved_nodes
    

    def query(self, query_str: str):
        retrieved_nodes: List[NodeWithScore] = self._retrieve_nodes(query_str)
        retrieved_nodes_text = "\n".join([node.text for node in retrieved_nodes])

        response_txt = self._llm.invoke(QUERY_PROMPT.format(
                                            context=retrieved_nodes_text,
                                            query_str=query_str
                                        )).content
        response = Response(response_txt, source_nodes=retrieved_nodes)

        return response 

# {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone'}
#  {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList'}
#   {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_@type': 'SowingTime', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_start_date': '15th June', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_end_date': '30th June'}
#   {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_has_seeding_rate_@type': 'schema:QuantitativeValue', 'has_growing_zones_has_seed_recommendations_has_seeding_rate_value': '55 kg per hectare'}
#   {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_needs_seed_sowing_spacing_@type': 'seedSpacing', 'has_growing_zones_has_seed_recommendations_needs_seed_sowing_spacing_value': '45 cm'}
#  {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_@type': 'SowingTime', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_start_date': '15th June', 'has_growing_zones_has_seed_recommendations_has_early_sowing_time_end_date': '30th June'}
#  {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_has_seeding_rate_@type': 'schema:QuantitativeValue', 'has_growing_zones_has_seed_recommendations_has_seeding_rate_value': '55 kg per hectare'}
#  {'@type': 'Crop', 'name': 'Soybean', 'has_growing_zones_@type': 'CropGrowingZone', 'has_growing_zones_name': 'North Eastern Hill zone', 'has_growing_zones_has_seed_recommendations_@type': 'SeedList', 'has_growing_zones_has_seed_recommendations_needs_seed_sowing_spacing_@type': 'seedSpacing', 'has_growing_zones_has_seed_recommendations_needs_seed_sowing_spacing_value': '45 cm'}
