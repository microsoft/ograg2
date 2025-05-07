from typing import Any, DefaultDict, Dict, List, Optional
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.langchain import LangChainLLM
from llama_index.packs.raptor import RaptorRetriever

MAX_TOKENS = 1024

QUERY_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Answer: """

class RaptorQueryEngine:
    def __init__(
        self,
        documents: List[BaseNode],
        llm,
        query_llm,
        embed_model,
        **kwargs: Any,
    ):
    
        # llm.metadata = LLMMetadata()
        self.retriever = RaptorRetriever(documents=documents, 
                                         llm=llm, #'gpt-4o', 
                                         embed_model=embed_model, #'text-embedding-3-small', 
                                         **kwargs
                        )
        self.llm = query_llm

    def _retrieve_nodes(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self.retriever.retrieve(query_str)
        return retrieved_nodes
    

    def query(self, query_str: str, return_context=False, rules=[], **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve_nodes(query_str)
        retrieved_nodes_text = "\n".join([node.text for node in retrieved_nodes])

        response_txt = self.llm.invoke(QUERY_PROMPT.format(
                                            context=retrieved_nodes_text + "\n".join(rules),
                                            query_str=query_str
                                        ), max_tokens=MAX_TOKENS).content
        response = Response(response_txt, source_nodes=retrieved_nodes)
        if return_context:
            return response, retrieved_nodes_text
        return response 

