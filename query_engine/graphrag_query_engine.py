from graphrag.query.cli import run_local_search, run_global_search
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from typing import Any, List, Dict

QUERY_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Answer: """

MAX_TOKENS = 4096

class GraphRAGQueryEngine:
    def __init__(
        self,
        llm: BaseLanguageModel,
        data_config: Dict[str, Any],
        method: str = "local",
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
        **kwargs: Any,
    ):
        self.data_config = data_config
        self.method = method
        self.community_level = community_level
        self.response_type = response_type
        self.llm = llm

    def _retrieve_nodes(self, query_str: str, return_context:bool=False, **kwargs):
        if self.method == "local":
            search_result = run_local_search(
                                data_dir=f"{self.data_config["documents_dir"]}/output/final/artifacts", 
                                root_dir=f"{self.data_config["documents_dir"]}", 
                                query=query_str,
                                community_level=self.community_level,
                                response_type=self.response_type,
                            )
        elif self.method == "global":
            search_result = run_global_search(
                                data_dir=f"{self.data_config["documents_dir"]}", 
                                root_dir=f"{self.data_config["documents_dir"]}/output/final/artifacts", 
                                query=query_str,
                                community_level=self.community_level,
                                response_type=self.response_type,
                            )

        return search_result.context_text
    
    def query(self, query_str: str, return_context=False, rules=[], **kwargs):
        retrieved_nodes_text = self._retrieve_nodes(query_str)

        response_txt = self.llm.invoke(QUERY_PROMPT.format(
                                            context=retrieved_nodes_text + "\n".join(rules),
                                            query_str=query_str
                                        ), max_tokens=MAX_TOKENS).content
        
        response = Response(response_txt)
        if return_context:
            return response, retrieved_nodes_text
        return response 
    

