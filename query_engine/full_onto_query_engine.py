from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
import os
import json

MAX_TOKENS = 1024

RAG_QUERY_PROMPT = """Given the context below, answer the following question. 
Note that the context is provided as a list of valid facts in a dictionary format. 
For example, if the context is {{'crop': 'soybean', 'seed': 'germination test requirements'}},
it means that the crop is soybean and its seed has germination test requirements.

Context: {context}

Question: {query_str}

Answer:
"""

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

class FullOntoQueryEngine:
    def __init__(self, llm: BaseLanguageModel, full_context: str = ""):
        self._llm = llm
        self.full_context = full_context
              
    @classmethod
    def from_ontology_path(
        cls,
        ontology_nodes_path: str,
        llm: BaseLanguageModel,
        embed_model: Embeddings=None
    ):
        nodes = load_graph_nodes(ontology_nodes_path)

        return cls(
            llm=llm,
            full_context=str(nodes)
        )
    
    def query(self, query_str: str, return_context: bool=False, **kwargs):
        response = self._llm.invoke(
            RAG_QUERY_PROMPT.format(
                context=self.full_context, 
                query_str=query_str
            ),
            max_tokens=MAX_TOKENS,
        )
        if return_context:
            return response, self.full_context
        return response
    