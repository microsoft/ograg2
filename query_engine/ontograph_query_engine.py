from typing import Any, DefaultDict, Dict, List, Optional, Union, Set
from llama_index.core.retrievers import BaseRetriever

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
from llama_index.core.node_parser import SimpleNodeParser
from utils import load_graph_nodes, load_graph_nodes_chunks, cosine_similarity, flatten_tree

MAX_TOKENS = 1024
# We first construct a tree based on the ontology mapping structure where each node is a dictionary
# The dictionary contains the key-value pairs of the node properties and their values
# x is a child of y if keys(x) is a subset of keys(y) and x[keys(x)] = y[keys(x)]
# Note that this makes it non-contiguous and we can have multiple parents for a node
# We can also have information structured in a multi-hop manner easy to retrieve
# 
# Can we filter the nodes based on the query string in a top-down manner? where we 
#   1. first vector search the nodes at the top-level (like "the crop is soybean" v/s "the crop is wheat")
#   2. then we go to the next level of nodes (like "the crop soybean has seed germination test requirements" v/s "the crop soybean has seed planting requirements")
#   3. and so on
#  We stop when the max embedding similarity is below a threshold between the nodes and the query string
#   or when we reach the leaf nodes of the ontology mapping
# 


# Write the corresponding text in the ontology node 

# Note that the context is provided as a list of valid facts in a dictionary format. 
# For example, if the context is {{'crop': 'soybean', 'seed': 'germination test requirements'}},
# it means that the crop is soybean and its seed has germination test requirements.

RAG_QUERY_PROMPT = """Given the context below, answer the following question. 
Note that the context is provided as a list of valid facts in a dictionary format and an optional set of rules.

Context: {context}

Question: {query_str}

Answer:
"""
# If the context does not help, you can ignore it and answer the question on your own.

class HyperNode: 
    def __init__(
        self, 
        key: str, 
        value: Union[str, List[str]], 
        embed_model: Embeddings, 
        edge_ids: List[int] = [], 
        key_embedding: Optional[np.ndarray] = None, 
        value_embedding: Optional[np.ndarray] = None, 
    ):
        self.key = key
        self.value = value
        if type(value) is str:
            if key_embedding is not None:
                self.key_embedding, self.value_embedding = key_embedding, value_embedding
            else:
                self.key_embedding, self.value_embedding = embed_model.embed_documents([key, value])
        elif type(value) is list:
            if key_embedding is not None:
                self.key_embedding, self.value_embedding = embed_model.embed_query(key), embed_model.embed_documents(value)
            else:
                self.key_embedding, self.value_embedding = key_embedding, value_embedding
        self.edge_ids = edge_ids

    def __str__(self):
        return f"{self.key}: {self.value}"

    def add_edge(self, edge_id: int):
        if edge_id not in self.edge_ids:
            self.edge_ids.append(edge_id)

    def similarity(self, query_embedding: np.ndarray, method='sum'):
        if method == 'sum':
            return cosine_similarity(query_embedding, self.key_embedding) + cosine_similarity(query_embedding, self.value_embedding)
        elif method == 'key_only':
            return cosine_similarity(query_embedding, self.key_embedding)
        elif method == 'value_only':
            return cosine_similarity(query_embedding, self.value_embedding)
        elif method == 'key_value_product':
            return cosine_similarity(query_embedding, self.key_embedding) * cosine_similarity(query_embedding, self.value_embedding)
        # return cosine_similarity(query_embedding, self.key_embedding) * (1 + cosine_similarity(query_embedding, self.value_embedding))
        
class HyperEdge:
    def __init__(self, nodes: List[HyperNode] = []):
        self.nodes = nodes

    def add_node(self, node: HyperNode):
        self.nodes.append(node)

    def to_dict(self):
        return {node.key: node.value for node in self.nodes}
    
    def to_text(self):
        return ', '.join([f"{node.key} '{node.value}'" for node in self.nodes])
    
class OntoHyperGraph:
    def __init__(
        self, 
        edges: List[HyperEdge], 
        nodes: List[HyperNode], 
        embed_model: Embeddings, 
        chunks: List[str] = None
    ):
        self.edges = edges
        self.nodes = nodes
        self.embed_model = embed_model
        self.chunks = chunks
        if chunks != None:
            assert (len(chunks) == len(edges))

    @classmethod
    def from_fact_lists(
        cls, 
        facts: List[Dict[str, Any]], 
        embed_model: Embeddings, 
        embeddings: Dict[str, np.ndarray] = None
    ):
        # facts are given as hyperedges
        # nodes = []
        nodes = {}
        hyperedges = []
        for i, fact in enumerate(facts):
            hyperedge_nodes = []
            for k, v in fact.items():
                if (k, v) in nodes:
                    nodes[(k, v)].add_edge(i)
                    hyperedge_nodes.append(nodes[(k, v)])
                else:
                    hypernode = HyperNode(
                                    k, v, embed_model, 
                                    key_embedding=embeddings[k] if embeddings is not None else None,
                                    value_embedding=embeddings[v] if embeddings is not None else None,
                                    edge_ids=[i]
                                )
                    nodes[(k, v)] = hypernode
                    hyperedge_nodes.append(hypernode)
            hyperedges.append(HyperEdge(hyperedge_nodes))

        return cls(
            nodes=list(nodes.values()),
            edges=hyperedges,
            embed_model=embed_model
        )
    
    def get_relevant_hyperedges(self, relevant_nodes: List[HyperNode], top_k: int=5):
        retrieved_node_to_edge_map = np.zeros((len(relevant_nodes), len(self.edges)))
        for i, node in enumerate(relevant_nodes):
            retrieved_node_to_edge_map[i, node.edge_ids] = 1

        nodes_covered = []
        relevant_edges = []
        while len(nodes_covered) < len(relevant_nodes) and len(relevant_edges) < top_k:
            nnodes_per_edge = retrieved_node_to_edge_map.sum(axis=0)
            max_id = max(np.where(nnodes_per_edge == nnodes_per_edge.max())[0], 
                         key=lambda x: len(self.edges[x].to_dict()))
            hyperedge = self.edges[max_id]
            nodes_covered_this = retrieved_node_to_edge_map[:, max_id].nonzero()[0].tolist()
            retrieved_node_to_edge_map[nodes_covered_this, :] = 0
            relevant_edges.append(hyperedge)
            nodes_covered += nodes_covered_this

        return relevant_edges
    
    def get_relevant_chunks (self, relevant_nodes: List[HyperNode], top_k: int=5):
        retrieved_node_to_edge_map = np.zeros((len(relevant_nodes), len(self.edges)))
        for i, node in enumerate(relevant_nodes):
            retrieved_node_to_edge_map[i, node.edge_ids] = 1

        nodes_covered = []
        relevant_chunks = set()
        while len(nodes_covered) < len(relevant_nodes) and len(relevant_chunks) < top_k:
            nnodes_per_edge = retrieved_node_to_edge_map.sum(axis=0)
            max_id = max(np.where(nnodes_per_edge == nnodes_per_edge.max())[0], key=lambda x: len(self.edges[x].to_dict()))
            nodes_covered_this = retrieved_node_to_edge_map[:, max_id].nonzero()[0].tolist()
            retrieved_node_to_edge_map[nodes_covered_this, :] = 0
            relevant_chunks.add(self.chunks[max_id])
            nodes_covered += nodes_covered_this

        return list(relevant_chunks)
    
    def select_nodes_attr(
            self, 
            sorted_nodes: List[HyperNode], 
            query_embedding: np.ndarray, 
            attr: str ='value', 
            es_node_steps: int = 20, 
            es_edge_steps: int = 5,
            es_maxnodes: int = 20
        ):
        selected_nodes, edges_covered = [], set()
        previous_nedges, no_new_nodes_for, no_new_edges_for = 0, 0, 0
        selected_attrs = []
        curr_sim = 0
        for i, node in enumerate(sorted_nodes):
            node_attr = getattr(node, attr)
            if node_attr in selected_attrs:
                no_new_nodes_for = 0
                selected_nodes.append(node)
                edges_covered = edges_covered.union(set(node.edge_ids))
                if len(edges_covered) == previous_nedges:
                    no_new_edges_for += 1
                else:
                    no_new_edges_for = 0
                continue
            new_text = ' '.join(selected_attrs + [node_attr] if node_attr not in selected_attrs else selected_attrs)
            new_embedding = self.embed_model.embed_query(new_text)
            # marginal gain
            if cosine_similarity(query_embedding, new_embedding) > curr_sim:
                no_new_nodes_for = 0
                curr_sim = cosine_similarity(query_embedding, new_embedding)
                selected_nodes.append(node)
                edges_covered = edges_covered.union(set(node.edge_ids))
                selected_attrs.append(node_attr)
                if len(edges_covered) == previous_nedges:
                    no_new_edges_for += 1
                else:
                    no_new_edges_for = 0
            else:
                no_new_nodes_for += 1
            if ((len(selected_nodes) == es_maxnodes) or (no_new_nodes_for == es_node_steps) or (no_new_edges_for == es_edge_steps)):
                break
            previous_nedges = len(edges_covered)
        return selected_nodes, edges_covered
    
    def get_relevant_hypernodes (self, query_embedding: np.ndarray, top_k: int=5):
        hypernodes_topkey = sorted(self.nodes, key=lambda x: x.similarity(query_embedding, method='key_only'), reverse=True)
        hypernodes_topvalue = sorted(self.nodes, key=lambda x: x.similarity(query_embedding, method='value_only'), reverse=True)
        # nodes_key, edges_covered1 = self.select_nodes_attr(hypernodes_topkey, query_embedding, attr='key')
        # # nodes_key = []
        # nodes_value, edges_covered2 = self.select_nodes_attr(hypernodes_topvalue, query_embedding, attr='value')
        return hypernodes_topkey[:top_k], hypernodes_topvalue[:top_k]

    
    def get_edge (self, edge_id: int):
        return self.edges[edge_id]
    
    def retrieve_context(self, query_str: str, nodes_top_k: int=10, top_k: int=5, context_length=1024):
        query_embedding = self.embed_model.embed_query(query_str)
        key_nodes, value_nodes = self.get_relevant_hypernodes(query_embedding, top_k=nodes_top_k)
        retrieved_nodes = key_nodes + value_nodes
        if self.chunks is not None:
            relevant_chunks = self.get_relevant_chunks(retrieved_nodes, top_k=top_k)
            relevant_context = '\n'.join(relevant_chunks)
        else:
            relevant_edges = self.get_relevant_hyperedges(retrieved_nodes, top_k=top_k)
            relevant_context = [edge.to_dict() for edge in relevant_edges]
            # relevant_context = [edge.to_text() for edge in relevant_edges]
        return retrieved_nodes, relevant_context
        # query_embedding = self.embed_model.embed_query(query_str)
        # hypernodes_topsim = sorted(self.nodes, key=lambda x: x.similarity(query_embedding), reverse=True)
        # retrieved_nodes = hypernodes_topsim[:top_k]

        # relevant_edges = self.get_relevant_hyperedges(retrieved_nodes)
        # relevant_context = [edge.to_dict() for edge in relevant_edges]



class OntoHyperGraphQueryEngine:
    def __init__(self, llm: BaseLanguageModel, onto_hypergraph: OntoHyperGraph, vector_retriever: BaseRetriever=None):
        self._llm = llm
        self._onto_hypergraph = onto_hypergraph
        self._vector_retriever = vector_retriever
              
    @classmethod
    def from_ontology_path(
        cls,
        ontology_nodes_path: str,
        llm: BaseLanguageModel,
        embed_model: Embeddings, 
        vector_retriever: BaseRetriever=None
    ):
        nodes = []
        for node in load_graph_nodes(ontology_nodes_path):
            nodes += flatten_tree(node)

        filtered_node_mappings = []
        for node in nodes:
            for node2 in filtered_node_mappings:
                if node == node2:
                    break
            else:
                filtered_node_mappings.append(node)
        
        nodes = filtered_node_mappings

        # if 'onto_hypernode_embeddings.pkl' not in os.listdir(ontology_nodes_path):
        nodes = [{k: str(v) for k, v in node.items()} for node in nodes]
        unique_texts = set()
        for node in nodes:
            for k, v in node.items():
                unique_texts.add(k)
                unique_texts.add(v)
        unique_texts = list(unique_texts)
        embeddings = embed_model.embed_documents(unique_texts)
        embeddings_dict = {text: np.array(emb) for text, emb in zip(unique_texts, embeddings)}
            # np.save(f'{ontology_nodes_path}/onto_hypernode_embeddings.pkl', embeddings)
            # pkl.dump(embeddings_dict, open(f'{ontology_nodes_path}/onto_hypernode_embeddings.pkl', 'wb'))
        
        # embeddings = pkl.load(open(f'{ontology_nodes_path}/onto_hypernode_embeddings.pkl', 'rb'))
        hypergraph = OntoHyperGraph.from_fact_lists(nodes, embed_model, embeddings=embeddings_dict)

        return cls(
            llm=llm,
            onto_hypergraph=hypergraph,
            vector_retriever=vector_retriever
        )
    
    @classmethod
    def from_ontology_path_and_documents(
        cls,
        ontology_nodes_path: str,
        documents: List[str],
        chunk_size: int,
        llm: BaseLanguageModel,
        embed_model: Embeddings
    ):
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        chunks = node_parser.get_nodes_from_documents(documents)
        nodes, mapped_chunks = [], []
        for node, mapped_chunk in zip(*load_graph_nodes_chunks(ontology_nodes_path, chunks)):
            flattened_nodes = flatten_tree(node)
            nodes += flattened_nodes
            mapped_chunks += [mapped_chunk] * len(flattened_nodes)

        filtered_node_mappings = []
        filtered_chunks = []
        for node, chunk in zip(nodes, mapped_chunks):
            for node2 in filtered_node_mappings:
                if node == node2:
                    break
            else:
                filtered_chunks.append(chunk)
                filtered_node_mappings.append(node)
        
        nodes = filtered_node_mappings

        if 'onto_hypernode_embeddings.npy' in os.listdir(ontology_nodes_path):
            embeddings = np.load(f'{ontology_nodes_path}/onto_hypernode_embeddings.npy')
            hypergraph = OntoHyperGraph.from_fact_lists(nodes, embed_model, embeddings=embeddings)
        else:
            hypergraph = OntoHyperGraph.from_fact_lists(nodes, embed_model,)
            embeddings = np.array([[hypernode.key_embedding, hypernode.value_embedding] for hypernode in hypergraph.nodes])
            np.save(f'{ontology_nodes_path}/onto_hypernode_embeddings.npy', embeddings)
        
        hypergraph.chunks = filtered_chunks

        return cls(
            llm=llm,
            onto_hypergraph=hypergraph
        )
    
    def query(self, query_str: str, top_k=5, context_length: int=1024, return_context: bool=False, rules=[], **kwargs):
        # kg_triples = self._triplet_retriever(query_str=query_str)
        
        # triples_node = TextNode(text=str(kg_triples), id_="kg_triples", 
        #                         metadata_template=query_str,)
        # triples_node = NodeWithScore(node=triples_node, score=999.0)
        # retrieved_nodes = [triples_node] + retrieved_nodes    
        # return retrieved_nodes
        _, relevant_context = self.retrieve_context(query_str, top_k=top_k, context_length=context_length)

        if self._vector_retriever is not None:
            relevant_context += [node.text for node in self._vector_retriever.retrieve(query_str)]
            response = self._llm.invoke(
                RAG_QUERY_PROMPT.format(
                    context=relevant_context + rules, 
                    query_str=query_str
                ),
                max_tokens=MAX_TOKENS
            )
        else:
            response = self._llm.invoke(
                RAG_QUERY_PROMPT.format(
                    context=relevant_context + rules, 
                    query_str=query_str
                ),
                max_tokens=MAX_TOKENS
            )
        if return_context:
            return response, relevant_context
        return response
    
    def retrieve_context(self, query_str, top_k=5, nodes_top_k: int=20, context_length: int=1024):
        retrieved_nodes, relevant_context = self._onto_hypergraph.retrieve_context(
            query_str, nodes_top_k=nodes_top_k, top_k=top_k, context_length=context_length)
        return retrieved_nodes, relevant_context
