from llama_index.core.prompts import PromptTemplate, PromptType
from typing import Any, DefaultDict, Dict, List, Optional
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

MAX_TOKENS = 1024

QUERY_PROMPT = """Answer the following question.
Question: {query_str}
---------------------
Answer: """

QUERY_RULE_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Answer: """

class LLMQueryEngine:
    def __init__(
        self,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ):

        self.llm = llm

    # async def query(self, query_str: str):
    def query(self, query_str: str, max_tokens=None, return_context=False, rules=[], **kwargs):
        # response = await self.llm.ainvoke(QUERY_PROMPT.format(query_str=query_str))
        if len(rules) == 0:
            response = self.llm.invoke(QUERY_PROMPT.format(query_str=query_str), 
                                       max_tokens=MAX_TOKENS if max_tokens is None else max_tokens)
        else:
            response = self.llm.invoke(QUERY_RULE_PROMPT.format(query_str=query_str, context="\n".join(rules)), 
                                       max_tokens=MAX_TOKENS if max_tokens is None else max_tokens)
        if return_context:
            return response, ""
        return response