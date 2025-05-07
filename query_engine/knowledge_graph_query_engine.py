""" Knowledge Graph Query Engine."""

import logging
from typing import Any, Dict, List, Optional, Sequence, DefaultDict
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.query_engine import BaseQueryEngine

from llama_index.core.prompts import PromptTemplate, PromptType
# from llama_index.core.prompts.mixin import DefaultDict[str, PromptType] # type: ignore
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import ServiceContext
from llama_index.core.utils import print_text
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import SimpleDirectoryReader

import sys, argparse

from typing import Sequence, Optional
from llama_index.core.storage.storage_context import StorageContext

try:
    from utils import (
        create_service_context,
        get_documents,
        create_or_load_index
    )
except:
    from ..utils import (
        create_service_context,
        get_documents,
        create_or_load_index
    )

from llama_index.core.retrievers import VectorIndexRetriever
import os
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.node_parser import SimpleNodeParser

MAX_TOKENS = 1024

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))  # Redirecting to stdout to see logs in Azure
HOME = '/home/ksartik/farmvibes-llm/agkgcopilot'
KG_STORAGE_PATH = '/home/ksartik/farmvibes-llm/agkgcopilot/kg/'
SOURCE_LENGTH = 10000

# Prompt
KG_QA_PROMPT_TMPL = (
    "Given the context information and not prior knowledge, "
    "answer the query below.\n"
    "----------------------\n"
    "The context information is provided below as triples for crop cultivation.\n"
    "The triples are organized as (subject, predicate, object).\n"
    "Context: \n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)


KG_QA_PROMPT = PromptTemplate(
    KG_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

KG_RETRIEVE_PROMPT_TMPL = (
    "The context information is provided below as triples for crop cultivation."
    "Given the context information and not prior knowledge, list all the relevant set of triples for the query below.\n"
    "List as many relevant triples as possible.\n"
    "Do not include any text other than the set of relevant triples.\n"
    "If the query does not match any triples, return an empty list. Do not add any other text\n"
    "--------------\n"
    "The triples are organized as (subject, predicate, object).\n"
    "Triples: \n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n\n"
    "Relevant Set of Triples:"
)


KG_RETRIEVE_PROMPT = PromptTemplate(
    KG_RETRIEVE_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)


TEXT_QA_PROMPT_TMPL = (
    "Given the following context, answer the question. Do not use any prior knowledge.\n"
    "The context is given as triples of a knowledge graph (subject, predicate, object).\n"
    "---------------------\n"
    "Context:\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}"
    "Answer: "
)
TEXT_QA_PROMPT = PromptTemplate(
    TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)



DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}.\n\n"
    "The existing answer is: {existing_answer}.\n\n"
    "There is an opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. \n"
    "If the context isn't useful, return the original answer.\n"
    "Make sure that the answer is relevant and confined to the original query."
    "Refined Answer: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

KG_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}.\n\n"
    "The existing answer is: {existing_answer}.\n\n"
    "There is an opportunity to refine and add to the existing answer (only if needed) with some more context below.\n"
    "The context information is provided below as triples for crop cultivation.\n"
    "The triples are organized as (subject, predicate, object).\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine and add to the original answer to better answer the query. \n"
    "If the context isn't useful, return the original answer.\n"
    "Make sure that the answer is relevant and confined to the original query."
    "Refined Answer: "
)
KG_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

DEFAULT_GRAPH_STORE_QUERY = """
MATCH (source)-[relation]->(target)
RETURN source, relation, target
"""


class KnowledgeTriplesGraphQueryEngine:
    """Original Knowledge graph query engine without subclassing the BaseQueryEngine.

    Query engine to call a knowledge graph triples as documents with an embedded vector index

    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        kg_triples_storage_path: Optional[str] = None,
        kg_qa_prompt: Optional[PromptTemplate] = None,
        kg_retrieve_prompt: Optional[PromptTemplate] = None,
        refine_prompt: Optional[PromptTemplate] = None,
        final_synthesis_prompt: Optional[PromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):

        # Get graph store query synthesis prompt
        self._kg_retrieve_prompt = (
            kg_retrieve_prompt
            or KG_RETRIEVE_PROMPT
        )
        
        self._kg_qa_prompt = (
            kg_qa_prompt
            or KG_QA_PROMPT
        )
        
        self._refine_prompt = (
            refine_prompt
            or DEFAULT_REFINE_PROMPT
        )
        
        self._final_synthesis_prompt = (
            final_synthesis_prompt
            or TEXT_QA_PROMPT
        )

        self.llm = llm
        self.embeddings = embeddings

        self._vector_retriever = vector_retriever

        self._verbose = verbose
        
        self._kg_triples_storage_path = (kg_triples_storage_path or KG_STORAGE_PATH)
        
        self._fmt_prompts = None
        
        self._doc_triples = self._load_triplets()
        
        self._retrieved_triples = None

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "kg_qa_prompt": self._kg_qa_prompt,
            "refine_prompt": self._refine_prompt,
            "final_synthesis_prompt": self._final_synthesis_prompt
        }

    def _update_prompts(self, prompts: DefaultDict[str, PromptType]) -> None:
        """Update prompts."""
        if "kg_qa_prompts" in prompts:
            self._kg_qa_prompt = prompts["kg_qa_prompt"]
        if "refine_prompt" in prompts:
            self._refine_prompt = prompts["refine_prompt"]
        if "final_synthesis_prompt" in prompts:
            self._final_synthesis_prompt = prompts["final_synthesis_prompt"]

    
    def _retrieve_nodes(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        
        kg_triples = self._triplet_retriever(query_str=query_str)
        
        triples_node = TextNode(text=str(kg_triples), id_="kg_triples", 
                                metadata_template=query_str,)
        triples_node = NodeWithScore(node=triples_node, score=999.0)
        retrieved_nodes = [triples_node] + retrieved_nodes    
        return retrieved_nodes
    
    def _triplet_retriever(self,
                        query_str: str
                        ) -> str:
        """ Extract relevant triples from each document"""
        
        retrieved_triples = []
        for idx, doc in enumerate(self._doc_triples):
            if self._verbose is True:
                print("Extracting relevant KG Triples")
                print(f"[Document {idx}]")
            
            fmt_prompt = self._kg_retrieve_prompt.format(
                context_str=doc.text, query_str=query_str
            )
            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            retrieved_triples.append(cur_response)
        
        self._retrieved_triples = retrieved_triples
        
        return ',\n'.join(retrieved_triples)
        
        

    def _synthesize(self, 
                    retrieved_nodes: List[NodeWithScore],
                    query_str: str
                    ) -> str:
        """Generate a response using create and refine strategy.

        The first node uses the 'QA' prompt.
        All subsequent nodes use the 'refine' prompt.

        """
        cur_response = None
        fmt_prompts = []
        for idx, node in enumerate(retrieved_nodes):
            if self._verbose is True:
                print("Querying the Retrieved Nodes")
                print(f"[Node {idx}]")
                display_source_node(node, source_length=SOURCE_LENGTH)
                pprint_source_node(node, source_length=SOURCE_LENGTH)
                
            context_str = node.get_content()
            if node.node_id == "kg_triples":
                print("Processing KG Triples")
                fmt_prompt = self._kg_qa_prompt.format(
                    context_str=context_str, query_str=query_str
                )
            else:
                fmt_prompt = self._refine_prompt.format(
                    context_msg=context_str,
                    query_str=query_str,
                    existing_answer=str(cur_response),
                )

            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            fmt_prompts.append(fmt_prompt)
        
        fmt_prompt = self._final_synthesis_prompt.format(context_str=str(cur_response), query_str=query_str)
        
        cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
        fmt_prompts.append(fmt_prompt)
        
        self._fmt_prompts = fmt_prompts

        return str(cur_response)

    def query(self, query_str: str, return_context: bool=False, **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve_nodes(query_str)

        response_txt = self._synthesize(
        retrieved_nodes,
        query_str
        )
            
        response = Response(response_txt, source_nodes=retrieved_nodes)

        if self._verbose:
            print_text(f"Final Response: {response}\n", color="green")
            
        if return_context:
            return response, self._retrieved_triples
        return response 
    
    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return None
    
    def _load_triplets(self):
        metadata_fn = lambda filename: {"file_name": filename, "type":"kg_triples"}
        doc_triples = SimpleDirectoryReader(self._kg_triples_storage_path, file_metadata=metadata_fn).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=8192)
        doc_triples = node_parser.get_nodes_from_documents(doc_triples)
        return doc_triples


class KnowledgeGraphListQueryEngineDefault:
    """Original Knowledge graph query engine without subclassing the BaseQueryEngine.

    Query engine to only call a knowledge graph in a list format

    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        storage_context: Optional[StorageContext] = None,
        kg_triples: Optional[List[Sequence[Any]]] = None,
        kg_qa_prompt: Optional[PromptTemplate] = None,
        refine_prompt: Optional[PromptTemplate] = None,
        final_synthesis_prompt: Optional[PromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):

        # Get graph store query synthesis prompt
        self._kg_qa_prompt = (
            kg_qa_prompt
            or KG_QA_PROMPT
        )
        
        self._refine_prompt = (
            refine_prompt
            or DEFAULT_REFINE_PROMPT
        )
        
        self._final_synthesis_prompt = (
            final_synthesis_prompt
            or TEXT_QA_PROMPT
        )

        self.llm = llm
        self.embeddings = embeddings

        self._verbose = verbose
        
        if kg_triples is None:
            assert storage_context is not None, "Must provide a storage context if no KG triplets provided."
            assert (
                storage_context.graph_store is not None
            ), "Must provide a graph store in the storage context."
            self._storage_context = storage_context
            self._graph_store = storage_context.graph_store
            self._graph_store_query = DEFAULT_GRAPH_STORE_QUERY
            self._kg_triples = self._extract_triplets()
            
        else:
            self._kg_triples = kg_triples
        
        self._fmt_prompts = None

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "kg_qa_prompt": self._kg_qa_prompt,
            "refine_prompt": self._refine_prompt,
            "final_synthesis_prompt": self._final_synthesis_prompt
        }

    def _update_prompts(self, prompts: DefaultDict[str, PromptType]) -> None:
        """Update prompts."""
        if "kg_qa_prompts" in prompts:
            self._kg_qa_prompt = prompts["kg_qa_prompt"]
        if "refine_prompt" in prompts:
            self._refine_prompt = prompts["refine_prompt"]
        if "final_synthesis_prompt" in prompts:
            self._final_synthesis_prompt = prompts["final_synthesis_prompt"]



    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        
        triples_node = TextNode(text=str(self._kg_triples), id_="kg_triples", 
                                metadata_template=query_str,)
        triples_node = NodeWithScore(node=triples_node, score=999.0)
        retrieved_nodes = [triples_node]    
        return retrieved_nodes

    def _synthesize(self, 
                    retrieved_nodes: List[NodeWithScore],
                    query_str: str
                    ) -> str:
        """Generate a response using create strategy.

        The first node uses the 'QA' prompt.
        """
        cur_response = None
        fmt_prompts = []
        for idx, node in enumerate(retrieved_nodes):
            if self._verbose is True:
                print(f"[Node {idx}]")
                display_source_node(node, source_length=SOURCE_LENGTH)
                pprint_source_node(node, source_length=SOURCE_LENGTH)
                
            context_str = node.get_content()
            if node.node_id == "kg_triples":
                print("Processing KG Triples")
                fmt_prompt = self._kg_qa_prompt.format(
                    context_str=context_str, query_str=query_str
                )

            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            fmt_prompts.append(fmt_prompt)
        
        self._fmt_prompts = fmt_prompts

        return str(cur_response)

    def query(self, query_str: str, **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve(query_str)

        response_txt = self._synthesize(
        retrieved_nodes,
        query_str
        )
            
        response = Response(response_txt, source_nodes=retrieved_nodes)

        if self._verbose:
            print_text(f"Final Response: {response}\n", color="green")
            
        return response 
    
    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return None   
    
    def _extract_triplets(self):
        graph_store_response = self._graph_store.query(query=self._graph_store_query)
        
        triplets = []

        for record in graph_store_response:
            # Assuming 'source', 'relation', and 'target' are the attributes
            source_node = record['source']
            relation = record['relation']
            target_node = record['target']

            # Convert to desired format, e.g., using node and relation properties
            triplet = (source_node['id'], relation[1], target_node['id'])
            triplets.append(triplet)
        
        return triplets
    
class KnowledgeGraphListQueryEngineOG:
    """Original Knowledge graph query engine without subclassing the BaseQueryEngine.

    Query engine to call a knowledge graph in a list format with an embedded vector index

    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        storage_context: Optional[StorageContext] = None,
        kg_triples: Optional[List[Sequence[Any]]] = None,
        kg_qa_prompt: Optional[PromptTemplate] = None,
        refine_prompt: Optional[PromptTemplate] = None,
        final_synthesis_prompt: Optional[PromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):

        # Get graph store query synthesis prompt
        self._kg_qa_prompt = (
            kg_qa_prompt
            or KG_QA_PROMPT
        )
        
        self._refine_prompt = (
            refine_prompt
            or DEFAULT_REFINE_PROMPT
        )
        
        self._final_synthesis_prompt = (
            final_synthesis_prompt
            or TEXT_QA_PROMPT
        )

        self.llm = llm
        self.embeddings = embeddings
        
        self._vector_retriever = vector_retriever

        self._verbose = verbose
        
        if kg_triples is None:
            assert storage_context is not None, "Must provide a storage context if no KG triplets provided."
            assert (
                storage_context.graph_store is not None
            ), "Must provide a graph store in the storage context."
            self._storage_context = storage_context
            self._graph_store = storage_context.graph_store
            self._graph_store_query = DEFAULT_GRAPH_STORE_QUERY
            self._kg_triples = self._extract_triplets()
            
        else:
            self._kg_triples = kg_triples
        
        self._fmt_prompts = None

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "kg_qa_prompt": self._kg_qa_prompt,
            "refine_prompt": self._refine_prompt,
            "final_synthesis_prompt": self._final_synthesis_prompt
        }

    def _update_prompts(self, prompts: DefaultDict[str, PromptType]) -> None:
        """Update prompts."""
        if "kg_qa_prompts" in prompts:
            self._kg_qa_prompt = prompts["kg_qa_prompt"]
        if "refine_prompt" in prompts:
            self._refine_prompt = prompts["refine_prompt"]
        if "final_synthesis_prompt" in prompts:
            self._final_synthesis_prompt = prompts["final_synthesis_prompt"]



    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        triples_node = TextNode(text=str(self._kg_triples), id_="kg_triples", 
                                metadata_template=query_str, )
        triples_node = NodeWithScore(node=triples_node, score=999.0)
        retrieved_nodes = [triples_node] + retrieved_nodes    
        return retrieved_nodes

    def _synthesize(self, 
                    retrieved_nodes: List[NodeWithScore],
                    query_str: str
                    ) -> str:
        """Generate a response using create and refine strategy.

        The first node uses the 'QA' prompt.
        All subsequent nodes use the 'refine' prompt.

        """
        cur_response = None
        fmt_prompts = []
        for idx, node in enumerate(retrieved_nodes):
            if self._verbose is True:
                print(f"[Node {idx}]")
                display_source_node(node, source_length=SOURCE_LENGTH)
                pprint_source_node(node, source_length=SOURCE_LENGTH)
                
            context_str = node.get_content()
            if node.node_id == "kg_triples":
                print("Processing KG Triples")
                fmt_prompt = self._kg_qa_prompt.format(
                    context_str=context_str, query_str=query_str
                )
            else:
                fmt_prompt = self._refine_prompt.format(
                    context_msg=context_str,
                    query_str=query_str,
                    existing_answer=str(cur_response),
                )

            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            fmt_prompts.append(fmt_prompt)
        
        fmt_prompt = self._final_synthesis_prompt.format(context_str=str(cur_response), query_str=query_str)
        
        cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
        fmt_prompts.append(fmt_prompt)
        
        self._fmt_prompts = fmt_prompts

        return str(cur_response)

    def query(self, query_str: str, **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve(query_str)

        response_txt = self._synthesize(
        retrieved_nodes,
        query_str
        )
            
        response = Response(response_txt, source_nodes=retrieved_nodes)

        if self._verbose:
            print_text(f"Final Response: {response}\n", color="green")
            
        return response 
    
    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return None   
    
    def _extract_triplets(self):
        graph_store_response = self._graph_store.query(query=self._graph_store_query)
        
        triplets = []

        for record in graph_store_response:
            # Assuming 'source', 'relation', and 'target' are the attributes
            source_node = record['source']
            relation = record['relation']
            target_node = record['target']

            # Convert to desired format, e.g., using node and relation properties
            triplet = (source_node['id'], relation[1], target_node['id'])
            triplets.append(triplet)
        
        return triplets
    
    
class KnowledgeGraphListQueryEngineReverse:
    """Query engine to call vector index first and then the knowledge graph

    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        storage_context: Optional[StorageContext] = None,
        kg_triples: Optional[List[Sequence[Any]]] = None,
        qa_prompt: Optional[PromptTemplate] = None,
        refine_prompt: Optional[PromptTemplate] = None,
        kg_refine_prompt: Optional[PromptTemplate] = None,
        final_synthesis_prompt: Optional[PromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):

        # Get graph store query synthesis prompt
        self._qa_prompt = (
            qa_prompt
            or TEXT_QA_PROMPT
        )
        
        self._refine_prompt = (
            refine_prompt
            or DEFAULT_REFINE_PROMPT
        )
        
        self._kg_refine_prompt = (
            kg_refine_prompt
            or KG_REFINE_PROMPT
        )
        
        self._final_synthesis_prompt = (
            final_synthesis_prompt
            or TEXT_QA_PROMPT
        )

        self.llm = llm
        self.embeddings = embeddings
        
        self._vector_retriever = vector_retriever

        self._verbose = verbose
        
        if kg_triples is None:
            assert storage_context is not None, "Must provide a storage context if no KG triplets provided."
            assert (
                storage_context.graph_store is not None
            ), "Must provide a graph store in the storage context."
            self._storage_context = storage_context
            self._graph_store = storage_context.graph_store
            self._graph_store_query = DEFAULT_GRAPH_STORE_QUERY
            self._kg_triples = self._extract_triplets()
            
        else:
            self._kg_triples = kg_triples
        
        self._fmt_prompts = None

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "qa_prompt": self._qa_prompt,
            "refine_prompt": self._refine_prompt,
            "kg_refine_prompt": self._kg_refine_prompt,
            "final_synthesis_prompt": self._final_synthesis_prompt
        }

    def _update_prompts(self, prompts: DefaultDict[str, PromptType]) -> None:
        """Update prompts."""
        if "qa_prompts" in prompts:
            self._qa_prompt = prompts["qa_prompt"]
        if "refine_prompt" in prompts:
            self._refine_prompt = prompts["refine_prompt"]
        if "kg_refine_prompt" in prompts:
            self._kg_refine_prompt = prompts["kg_refine_prompt"]
        if "final_synthesis_prompt" in prompts:
            self._final_synthesis_prompt = prompts["final_synthesis_prompt"]



    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        triples_node = TextNode(text=str(self._kg_triples), id_="kg_triples", 
                                metadata_template=query_str, )
        triples_node = NodeWithScore(node=triples_node, score=999.0)
        retrieved_nodes = retrieved_nodes + [triples_node]    
        return retrieved_nodes

    def _synthesize(self, 
                    retrieved_nodes: List[NodeWithScore],
                    query_str: str
                    ) -> str:
        """Generate a response using create and refine strategy.

        The first node uses the 'QA' prompt.
        All subsequent nodes use the 'refine' prompt.

        """
        cur_response = None
        fmt_prompts = []
        for idx, node in enumerate(retrieved_nodes):
            if self._verbose is True:
                print(f"[Node {idx}]")
                display_source_node(node, source_length=SOURCE_LENGTH)
                pprint_source_node(node, source_length=SOURCE_LENGTH)
                
            context_str = node.get_content()
            
            if idx == 0:
                fmt_prompt = self._qa_prompt.format(context_str=str(cur_response), query_str=query_str)
            
            if node.node_id == "kg_triples":
                print("Processing KG Triples")
                fmt_prompt = self._kg_refine_prompt.format(
                    context_msg=context_str,
                    query_str=query_str,
                    existing_answer=str(cur_response),
                )
            else:
                fmt_prompt = self._refine_prompt.format(
                context_msg=context_str,
                query_str=query_str,
                existing_answer=str(cur_response)
                )
                    

            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            fmt_prompts.append(fmt_prompt)
        
        fmt_prompt = self._final_synthesis_prompt.format(context_str=str(cur_response), query_str=query_str)
        
        cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
        fmt_prompts.append(fmt_prompt)
        
        self._fmt_prompts = fmt_prompts

        return str(cur_response)

    def query(self, query_str: str, **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve(query_str)

        response_txt = self._synthesize(
        retrieved_nodes,
        query_str
        )
            
        response = Response(response_txt, source_nodes=retrieved_nodes)

        if self._verbose:
            print_text(f"Final Response: {response}\n", color="green")
            
        return response 
    
    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return None   
    
    def _generate_triplets(self):
        graph_store_response = self._graph_store.query(query=self._graph_store_query)
        
        triplets = []

        for record in graph_store_response:
            # Assuming 'source', 'relation', and 'target' are the attributes
            source_node = record['source']
            relation = record['relation']
            target_node = record['target']

            # Convert to desired format, e.g., using node and relation properties
            triplet = (source_node['id'], relation[1], target_node['id'])
            triplets.append(triplet)
        
        return triplets
    
    def _extract_triplets(self):
        raise NotImplementedError


class KnowledgeGraphListQueryEngine(BaseQueryEngine):
    """Knowledge graph query engine.

    Query engine to call a knowledge graph in a list format with an embedded vector index

    Args:
        service_context (Optional[ServiceContext]): A service context to use.
        verbose (bool): Whether to print intermediate results.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        kg_triples: Optional[List[Sequence[Any]]] = None,
        kg_qa_prompt: Optional[PromptTemplate] = None,
        refine_prompt: Optional[PromptTemplate] = None,
        final_synthesis_prompt: Optional[PromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):

        # Get graph store query synthesis prompt
        self._kg_qa_prompt = (
            kg_qa_prompt
            or KG_QA_PROMPT
        )
        
        self._refine_prompt = (
            refine_prompt
            or DEFAULT_REFINE_PROMPT
        )
        
        self._final_synthesis_prompt = (
            final_synthesis_prompt
            or TEXT_QA_PROMPT
        )

        self.llm = llm
        self.embeddings = embeddings
        
        self._vector_retriever = vector_retriever

        self._verbose = verbose
        
        self._kg_triples = kg_triples

        super().__init__(self._service_context.callback_manager)
        
        self._fmt_prompts = None

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "kg_qa_prompt": self._kg_qa_prompt,
            "refine_prompt": self._refine_prompt,
            "final_synthesis_prompt": self._final_synthesis_prompt
        }

    def _update_prompts(self, prompts: DefaultDict[str, PromptType]) -> None:
        """Update prompts."""
        if "kg_qa_prompts" in prompts:
            self._kg_qa_prompt = prompts["kg_qa_prompt"]
        if "refine_prompt" in prompts:
            self._refine_prompt = prompts["refine_prompt"]
        if "final_synthesis_prompt" in prompts:
            self._final_synthesis_prompt = prompts["final_synthesis_prompt"]



    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        triples_node = TextNode(text=str(self._kg_triples), id_="kg_triples", 
                                metadata_template=query_str, )
        triples_node = NodeWithScore(node=triples_node, score=999.0)
        retrieved_nodes = [triples_node] + retrieved_nodes    
        return retrieved_nodes

    def _synthesize(self, 
                    retrieved_nodes: List[NodeWithScore],
                    query_str: str
                    ) -> str:
        """Generate a response using create and refine strategy.

        The first node uses the 'QA' prompt.
        All subsequent nodes use the 'refine' prompt.

        """
        cur_response = None
        fmt_prompts = []
        for idx, node in enumerate(retrieved_nodes):
            if self._verbose is True:
                print(f"[Node {idx}]")
                display_source_node(node, source_length=SOURCE_LENGTH)
                pprint_source_node(node, source_length=SOURCE_LENGTH)
                
            context_str = node.get_content()
            if node.node_id == "kg_triples":
                print("Processing KG Triples")
                fmt_prompt = self._kg_qa_prompt.format(
                    context_str=context_str, query_str=query_str
                )
            else:
                fmt_prompt = self._refine_prompt.format(
                    context_msg=context_str,
                    query_str=query_str,
                    existing_answer=str(cur_response),
                )

            cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
            fmt_prompts.append(fmt_prompt)
        
        fmt_prompt = self._final_synthesis_prompt.format(context_str=str(cur_response), query_str=query_str)
        
        cur_response = self.llm.invoke(fmt_prompt, max_tokens=MAX_TOKENS).content
        fmt_prompts.append(fmt_prompt)
        
        self._fmt_prompts = fmt_prompts

        return str(cur_response)
        
        
        
    def _query(self, query_str: str, **kwargs) -> RESPONSE_TYPE:
        """Query the hybrid knowledge graph."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_str}
        ) as query_event:
            retrieved_nodes: List[NodeWithScore] = self._retrieve(query_str)

        response_txt = self._synthesize(
        retrieved_nodes,
        query_str
        )
            
        response = Response(response_txt, source_nodes=retrieved_nodes)

        if self._verbose:
            print_text(f"Final Response: {response}\n", color="green")

        #query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
    
    def get_llm_prompts(self) -> Optional[List[Any]]:
        return self._fmt_prompts
    
    async def _aquery(self, query_str:str):
        """Query the graph store."""

        return None
    
    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return None
    

def main(
    kg_storage_path: str, 
    service_context: ServiceContext, 
    documents_dir: str,
    index_dir: str,
    output_dir: str,
    connection_id: str
) -> None:
    
    documents = get_documents(documents_dir)
    vector_index = create_or_load_index(index_directory=index_dir, service_context=service_context, documents=documents)
    vector_retriever = VectorIndexRetriever(index=vector_index)
    kg_query_engine_triples = KnowledgeTriplesGraphQueryEngine(service_context=service_context, kg_triples_storage_path=kg_storage_path,
                                                                vector_retriever=vector_retriever, verbose=True)
    
    while True:
        query = input("Type your query (press Enter to stop):")
        if query == '':
            break
        response = kg_query_engine_triples.query(query_str=query)
        print (f'Response: {response}')


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Question answering using knowledge graph augmented context.")
        parser.add_argument(
            "-kg",
            "--kg_storage_path",
            required=True,
            # default=f'{HOME}/data/ontology/',
            help="Path to the directory of the knowledge graph.",
        )
        parser.add_argument(
            "-d",
            "--documents_dir",
            required=True,
            # default=f'{HOME}/data/md/',
            help="Input directory containing the markdown files to be processed.",
        )
        parser.add_argument(
            "-id",
            "--index_dir",
            # required=True,
            default=None, 
            help="Index directory to store the vector index.",
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
        if args.index_dir is None:
            kg_name = os.path.basename(args.kg_storage_path) if '/' not in args.kg_storage_path else os.path.basename(args.kg_storage_path[:-1])
            index_dir = f"{HOME}/vector_{kg_name}"
            

        service_context = create_service_context(args.connection_id, embedding_model, args.deployment_name)

        main(
            kg_storage_path=args.kg_storage_path,
            service_context=service_context,
            documents_dir=args.documents_dir,
            index_dir=index_dir,
            output_dir=args.output_dir,
            connection_id=args.connection_id
        )

    except Exception as e:
        LOGGER.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
