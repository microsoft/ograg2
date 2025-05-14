from llama_index.core.prompts import PromptTemplate, PromptType
from typing import Any, DefaultDict, Dict, List, Optional, Union
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode
import json
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
from llama_index.core.node_parser import SimpleNodeParser

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


RAG_QUERY_PROMPT = """Given the context below, answer the following question. 
Note that the context is provided as a list of valid facts in a dictionary format. 
Context: {context}
Question: {query_str}
Answer:
"""
# If the context does not help, you can ignore it and answer the question on your own.

def cosine_similarity (x, y):
    if type(y[0]) is list:
        sims = []
        for yi in y: sims.append(cosine_similarity(x, yi))
        return sims.max()
    return np.dot(np.array(x), np.array(y))/(np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(y)))


def flatten_tree (node):
    node_type = ''
    for k, v in node.items():
        if '@type' in k:
            node_type = v
            break
    node_context = {f'{node_type} {k}': v for k, v in node.items() if not isinstance(v, dict) and not isinstance(v, list) and not k.startswith('@')}
    flattened_nodes = [node_context]
    for k, v in node.items():
        if '@type' in k:
            continue
        elif isinstance(v, dict):
            for v_node in flatten_tree(v):
                v_node = {f'{node_type} {k} {k2}': v2 for k2, v2 in v_node.items()}
                flattened_nodes.append({**node_context, **v_node})
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for v_node in flatten_tree(item):
                        v_node = {f'{node_type} {k} {k2}': v2 for k2, v2 in v_node.items()}
                        flattened_nodes.append({**node_context, **v_node})
                else:
                    flattened_nodes.append({**node_context, k: item})
    return flattened_nodes

def flatten_tree_single (node):
    node_type = ''
    for k, v in node.items():
        if '@type' in k:
            node_type = v
            break
    # node_context = {f'{node_type} {k}': v for k, v in node.items() if not isinstance(v, dict) and not isinstance(v, list) and not k.startswith('@')}
    # flattened_nodes = [node_context]
    flattened_node = {}
    for k, v in node.items():
        if '@type' in k:
            continue
        elif isinstance(v, dict):
            for k2, v2 in flatten_tree(v).items():
                flattened_node[f'{node_type} {k} {k2}'] = v2
        else:
            flattened_node[f'{node_type} {k}'] = v
    return flattened_node

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

def load_graph_nodes_chunks(ontology_nodes_path, chunks):
    nodes = []
    mapped_chunks = []
    for ontdir in os.listdir(ontology_nodes_path):
        ontdir = os.path.join(ontology_nodes_path, ontdir)
        if os.path.isdir(ontdir) and ontdir.endswith('ontology'):
            for fname in os.listdir(ontdir):
                if fname.endswith('.jsonld'):
                    fname = os.path.join(ontdir, fname)
                    node_id = int(fname.split('ontology_node_')[1].split('.jsonld')[0])
                    with open(f'{fname}', 'r') as f:
                        nodes += json.load(f)['@graph']
                        mapped_chunks.append(chunks[node_id].text)
    return nodes, mapped_chunks

class OntoTree: 
    def __init__(self, data: Dict[str, Any], embed_model: Embeddings):
        self.data = data
        self.embed_model = embed_model
        self.children = []
        self.parents = []
        self.level = 0

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)

    def __str__(self):
        s = ' ' * self.level + self.data.__str__()
        for child in self.children:
            child.level = self.level + 1
            s += '\n' + child.__str__()
        return s
    
    def _compress(self):
        for child in self.children:
            new_parents = set(child.parents).difference(self.parents)
            redundant_parents = set(child.parents).intersection(self.parents)
            child.parents = new_parents
            for parent in redundant_parents:
                parent.children.remove(child)
            child._compress()
    
    def depth(self):
        if self.children == []:
            return 1
        else:
            return 1 + max([child.depth() for child in self.children])
    
    def num_nodes(self):
        if self.children == []:
            return 1
        else:
            return 1 + sum([child.num_nodes() for child in self.children])
        
    def _embed_tree(self):
        # print (f"Embedding tree with data: {self.data}")
        self.embedded_data = np.array(list(zip(
            self.embed_model.embed_documents(list(self.data.keys())), 
            self.embed_model.embed_documents(list(self.data.values()))
        )))
        for child in self.children:
            child._embed_tree()

    def similarity_root(self, query_embedding):
        similarity = 0
        query_embedding = np.array(query_embedding)
        data_keys = list(self.data.keys())
        for i in range(self.embedded_data.shape[0]):
            if data_keys[i].startswith("@"):
                similarity += cosine_similarity(query_embedding, self.embedded_data[i, 1, :])
            else:
                similarity += cosine_similarity(query_embedding, self.embedded_data[i, 0, :]) * \
                                (1 + cosine_similarity(query_embedding, self.embedded_data[i, 1, :]))
                # similarity += cosine_similarity(query_embedding, self.embedded_data[i, 0, :]) + \
                #                 cosine_similarity(query_embedding, self.embedded_data[i, 1, :])
            # similarity += self.embed_model.similarity(query_embedding, i[0]) * (1 + self.embed_model.similarity(query_embedding, i[1]))
        similarity = similarity #/nsims #/(self.embedded_data.shape[0]*2)
        return similarity
    
    def _full_tree(self):
        nodes = [self.data]
        for child in self.children:
            nodes += child._full_tree()
        return nodes
    
    def set_embeddings(self, embeddings):
        self.embedded_data = embeddings[0]
        for i, child in enumerate(self.children):
            child.set_embeddings(embeddings[i+1])
    
    def get_concat_embeddings(self):
        embeddings = [self.embedded_data]
        for child in self.children:
            embeddings.append(child.get_concat_embeddings())
        return embeddings

class OntoGraph:
    def __init__(self, trees: List[OntoTree], embed_model: Embeddings, embeddings: Optional[np.ndarray] = None,) -> None:
        self.trees = trees
        self.embed_model = embed_model
        self._compress_trees()
        if embeddings is None:
            self._embed_trees()
        else:
            for i, tree in enumerate(self.trees):
                tree.set_embeddings(embeddings[i])

    @classmethod
    def from_node_mappings(
        cls, 
        node_mappings: List[Dict[str, Any]], 
        embed_model: Embeddings,
        embeddings: Optional[np.ndarray] = None,
    ):
        trees = []
        for node in node_mappings:
            trees.append(OntoTree(node, embed_model=embed_model))

        for i, node in enumerate(node_mappings):
            for j, node2 in enumerate(node_mappings):
                if (node != node2) and (set(node.keys()).issubset(set(node2.keys()))):
                    for k, v in node.items():
                        if node2[k] != v:
                            break
                    else:
                        trees[i].add_child(trees[j])
                        trees[j].add_parent(trees[i])
        return cls(
            trees=[tree for tree in trees if tree.parents == []],
            embed_model=embed_model,    
            embeddings=embeddings
        )
    
    def _compress_trees(self):
        for tree in self.trees:
            tree._compress()

    def _embed_trees(self):
        for tree in tqdm(self.trees, desc="Embedding trees"):
            tree._embed_tree()
            # print(tree.embedded_data)

    def _retrieve_nodes(self, query_str: str, threshold_sim=0.75, context_length=1024) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        query_embedding = self.embed_model.embed_query(query_str)

        retrieved_nodes = []
        nodes = self.trees

        max_similarity = 0
        curr_nodes = nodes.copy()
        while (len(curr_nodes) != 0):
            next_nodes = []
            for node in curr_nodes:
                similarity = node.similarity_root(query_embedding)
                max_similarity = max(max_similarity, similarity)
                if similarity > threshold_sim:
                    # if context length reached, then search within the nodes further.
                    retrieved_nodes += node._full_tree()
                else:
                    next_nodes += node.children
            curr_nodes = next_nodes

        return retrieved_nodes
    
    def get_concat_embeddings(self):
        return [tree.get_concat_embeddings() for tree in self.trees]
    
class OntoGraphQueryEngine:
    def __init__(self, llm: BaseLanguageModel, onto_graph: OntoGraph):
        self._llm = llm
        self._onto_graph = onto_graph
              
    @classmethod
    def from_ontology_path(
        cls,
        ontology_nodes_path: str,
        llm: BaseLanguageModel,
        embed_model: Embeddings
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

        if 'onto_embeddings.pkl' in os.listdir(ontology_nodes_path):
            embeddings = pkl.load(open(f'{ontology_nodes_path}/onto_embeddings.pkl', 'rb'))
            onto_graph = OntoGraph.from_node_mappings(nodes, embed_model, embeddings=embeddings)
        else:
            onto_graph = OntoGraph.from_node_mappings(nodes, embed_model)
            embeddings = onto_graph.get_concat_embeddings()
            pkl.dump(embeddings, open(f'{ontology_nodes_path}/onto_embeddings.pkl', 'wb'))
        return cls(
            llm=llm,
            onto_graph=onto_graph
        )
    
    def query(self, query_str: str, threshold_sim: int=0.75, context_length: int=1024, **kwargs):
        retrieved_nodes_info = self._onto_graph._retrieve_nodes(query_str, threshold_sim=threshold_sim)
        response_text = self._llm.invoke(
            RAG_QUERY_PROMPT.format(
                context=retrieved_nodes_info, 
                query_str=query_str
            ),
            max_tokens=MAX_TOKENS
        ).content
        return response_text

# We can also construct a hypergraph where each node is a tuple of key-value pairs
class HyperNode: 
    def __init__(self, key: str, value: Union[str, List[str]], embed_model: Embeddings, edge_ids: List[int] = [], embeddings: Optional[np.ndarray] = None):
        self.key = key
        self.value = value
        if type(value) is str:
            if embeddings is not None:
                self.key_embedding, self.value_embedding = embeddings[0], embeddings[1]
            else:
                self.key_embedding, self.value_embedding = embed_model.embed_documents([key, value])
        elif type(value) is list:
            if embeddings is not None:
                self.key_embedding, self.value_embedding = embed_model.embed_query(key), embed_model.embed_documents(value)
            else:
                self.key_embedding, self.value_embedding = embeddings[0], embeddings[1:]
        self.edge_ids = edge_ids

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

class OntoHyperGraph:
    def __init__(self, edges: List[HyperEdge], nodes: List[HyperNode], embed_model: Embeddings, chunks: List[str] = None):
        self.edges = edges
        self.nodes = nodes
        self.embed_model = embed_model
        if chunks != None:
            assert (len(chunks) == len(edges))
            self.chunks = chunks

    @classmethod
    def from_fact_lists(
        cls, 
        facts: List[Dict[str, Any]], 
        embed_model: Embeddings, 
        embeddings: Optional[np.ndarray] = None
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
                                    embeddings=embeddings[len(nodes)] if embeddings is not None else None, 
                                    edge_ids=[i]
                                )
                    nodes[(k, v)] = hypernode
                    hyperedge_nodes.append(hypernode)
            hyperedges.append(HyperEdge(hyperedge_nodes))

        nodes = list(nodes.values())

        return cls(
            nodes=nodes,
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
            max_id = max(np.where(nnodes_per_edge == nnodes_per_edge.max())[0], key=lambda x: len(self.edges[i].to_dict()))
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
        relevant_chunks = []
        while len(nodes_covered) < len(relevant_nodes) and len(relevant_chunks) < top_k:
            nnodes_per_edge = retrieved_node_to_edge_map.sum(axis=0)
            max_id = max(np.where(nnodes_per_edge == nnodes_per_edge.max())[0], key=lambda x: len(self.edges[i].to_dict()))
            nodes_covered_this = retrieved_node_to_edge_map[:, max_id].nonzero()[0].tolist()
            retrieved_node_to_edge_map[nodes_covered_this, :] = 0
            relevant_chunks.append(self.chunks[max_id])
            nodes_covered += nodes_covered_this

        return relevant_chunks
    
    def select_nodes_attr(self, sorted_nodes, query_embedding, attr='value', es_node_steps=20, es_edge_steps=5, es_maxnodes=20):
        selected_nodes, edges_covered = [], set()
        previous_nedges, no_new_nodes_for, no_new_edges_for = 0, 0, 0
        selected_text = ''
        curr_sim = 0
        for i, node in enumerate(sorted_nodes):
            new_text = selected_text + ' ' + getattr(node, attr) if len(selected_text) > 0 else getattr(node, attr)
            new_embedding = self.embed_model.embed_query(new_text)
            # marginal gain
            if cosine_similarity(query_embedding, new_embedding) > curr_sim:
                no_new_nodes_for = 0
                curr_sim = cosine_similarity(query_embedding, new_embedding)
                selected_nodes.append(node)
                edges_covered = edges_covered.union(set(node.edge_ids))
                selected_text = new_text
                if len(edges_covered) == previous_nedges:
                    no_new_edges_for += 1
                else:
                    no_new_edges_for = 0
            else:
                no_new_nodes_for += 1
            if (len(selected_nodes) == es_maxnodes) or (no_new_nodes_for == es_node_steps) or (no_new_edges_for == es_edge_steps):
                break
            previous_nedges = len(edges_covered)
        return selected_nodes, edges_covered
    
    def get_relevant_hypernodes (self, query_embedding):
        hypernodes_topkey = sorted(self.nodes, key=lambda x: x.similarity(query_embedding, method='key_only'), reverse=True)
        hypernodes_topvalue = sorted(self.nodes, key=lambda x: x.similarity(query_embedding, method='value_only'), reverse=True)
        nodes_key, _ = self.select_nodes_attr(hypernodes_topkey, query_embedding, attr='key')
        nodes_value, _ = self.select_nodes_attr(hypernodes_topvalue, query_embedding, attr='value')

        return nodes_key, nodes_value

    
    def get_edge (self, edge_id: int):
        return self.edges[edge_id]
    
    def set_node_edges (self):
        for node in self.nodes:
            node.edges = [self.edges[i] for i in node.edge_ids]
    
    def retrieve_context(self, query_str: str, top_k: int=5, context_length=1024):
        query_embedding = self.embed_model.embed_query(query_str)
        key_nodes, value_nodes = self.get_relevant_hypernodes(query_embedding)
        retrieved_nodes = key_nodes + value_nodes
        if self.chunks is not None:
            relevant_edges = self.get_relevant_chunks(retrieved_nodes, top_k=top_k)
            relevant_context = '\n'.join(relevant_edges)
        else:
            relevant_edges = self.get_relevant_hyperedges(retrieved_nodes, top_k=top_k)
            relevant_context = [edge.to_dict() for edge in relevant_edges]
        return retrieved_nodes, relevant_context
        # query_embedding = self.embed_model.embed_query(query_str)
        # hypernodes_topsim = sorted(self.nodes, key=lambda x: x.similarity(query_embedding), reverse=True)
        # retrieved_nodes = hypernodes_topsim[:top_k]

        # relevant_edges = self.get_relevant_hyperedges(retrieved_nodes)
        # relevant_context = [edge.to_dict() for edge in relevant_edges]



class OntoHyperGraphQueryEngine:
    def __init__(self, llm: BaseLanguageModel, onto_hypergraph: OntoHyperGraph):
        self._llm = llm
        self._onto_hypergraph = onto_hypergraph
              
    @classmethod
    def from_ontology_path(
        cls,
        ontology_nodes_path: str,
        llm: BaseLanguageModel,
        embed_model: Embeddings
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

        if 'onto_hypernode_embeddings.npy' in os.listdir(ontology_nodes_path):
            embeddings = np.load(f'{ontology_nodes_path}/onto_hypernode_embeddings.npy')
            hypergraph = OntoHyperGraph.from_fact_lists(nodes, embed_model, embeddings=embeddings)
        else:
            hypergraph = OntoHyperGraph.from_fact_lists(nodes, embed_model,)
            embeddings = np.array([[hypernode.key_embedding, hypernode.value_embedding] for hypernode in hypergraph.nodes])
            np.save(f'{ontology_nodes_path}/onto_hypernode_embeddings.npy', embeddings)
        
        return cls(
            llm=llm,
            onto_hypergraph=hypergraph
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
    
    def query(self, query_str: str, top_k=5, context_length: int=1024, **kwargs):
        retrieved_nodes, relevant_context = self._onto_hypergraph.retrieve_context(query_str, top_k=top_k, context_length=context_length)
        response = self._llm.invoke(
            RAG_QUERY_PROMPT.format(
                context=str(relevant_context), 
                query_str=query_str
            ),
            max_tokens=MAX_TOKENS
        )
        return response
    
    def retrieve_context(self, query_str, top_k=5, context_length: int=1024):
        retrieved_nodes, relevant_context = self._onto_hypergraph.retrieve_context(query_str, top_k=top_k, context_length=context_length)
        return retrieved_nodes, relevant_context



# [
#     {'Crop name': 'Soybean', 
#         'Crop seed_germination_test_requirements_are': 'Seed Germination Test Requirements', 
#         'Crop harvesting_guidelines_are': 'Harvesting Guidelines', 
#         'Crop storage_guidelines_are': 'Storage Guidelines', 
#         'Crop has_growing_zones CropGrowingZones CropGrowingZone  name': 'Central Zone', 
#         'Crop has_growing_zones CropGrowingZones CropGrowingZone  has_seed_recommendations variety_name': 'JS 20-116 (2019)'}, 
#     {'cropCult:Crop name': 'Soybean', 
#         'cropCult:Crop seed_germination_test_requirements_are': 'Germination percentage should be above 70%', 
#         'cropCult:Crop harvesting_guidelines_are': 'Harvest when 85-90% pods turn brown and start to split', 
#         'cropCult:Crop storage_guidelines_are': 'Store in cool and dry place', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  name': 'Central Zone', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  has_seed_recommendations variety_name': 'JS 20-69 (2016)'}, 
#     {'Crop name': 'Soybean', 
#         'Crop seed_germination_test_requirements_are': 'Seed Germination Test Requirements', 
#         'Crop harvesting_guidelines_are': 'Harvesting Guidelines', 
#         'Crop storage_guidelines_are': 'Storage Guidelines', 
#         'Crop has_growing_zones CropGrowingZones CropGrowingZone  name': 'Central Zone', 
#         'Crop has_growing_zones CropGrowingZones CropGrowingZone  has_seed_recommendations variety_name': 'JS 20-94 (2019)'}, 
#     {'Crop name': 'Soybean', 
#         'Crop has_growing_zones CropGrowingZone name': 'Central zone'}, 
#     {'cropCult:Crop name': 'Soybean', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  name': 'Central zone'}, 
#     {'cropCult:Crop name': 'Soybean', 
#         'cropCult:Crop seed_germination_test_requirements_are': 'Germination percentage should be above 70%', 
#         'cropCult:Crop harvesting_guidelines_are': 'Harvest when 85-90% pods turn brown and start to split', 
#         'cropCult:Crop storage_guidelines_are': 'Store in cool and dry place', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  name': 'Central Zone', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  has_seed_recommendations cropCult:SeedList has_fertilizer_application_requirements cropCult:Fertilization nutrient_name': 'Nitrogen, Phosphorus, Potassium', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  has_seed_recommendations cropCult:SeedList has_fertilizer_application_requirements cropCult:Fertilization quantity': '20-60 kg/ha', 
#         'cropCult:Crop has_growing_zones cropCult:CropGrowingZones CropGrowingZone  has_seed_recommendations cropCult:SeedList has_fertilizer_application_requirements cropCult:Fertilization stage_of_application': 'At sowing'}, 
#     {'cropCult:SeedList variety_name': 'Insect resistant/tolerant variety'}, 
#     {'Crop name': 'Soybean', 
#         'Crop seed_germination_test_requirements_are': 'Seeds should have a germination rate of at least 85%', 
#         'Crop harvesting_guidelines_are': 'Harvest when 85-90% pods turn brown and start to split', 
#         'Crop storage_guidelines_are': 'Store in a cool, dry place', 
#         'Crop has_growing_zones CropGrowingZones name': 'Zone 1', 
#         'Crop has_growing_zones CropGrowingZones has_seed_recommendations SeedList has_fertilizer_application_requirements Fertilization nutrient_name': 'Nitrogen', 
#         'Crop has_growing_zones CropGrowingZones has_seed_recommendations SeedList has_fertilizer_application_requirements Fertilization quantity': '50 kg/ha', 
#         'Crop has_growing_zones CropGrowingZones has_seed_recommendations SeedList has_fertilizer_application_requirements Fertilization stage_of_application': 'At sowing'}, 
#     {'cropCult:IrrigationRequirement quantity': 'Lifesaving irrigation during critical stages', 
#         'cropCult:IrrigationRequirement stage_of_application': 'Seedling, flowering and pod filling stages'}, 
#     {'Crop name': 'Wheat', 
#         'Crop seed_germination_test_requirements_are': '95% germination rate', 
#         'Crop harvesting_guidelines_are': 'Harvest when moisture content is below 14%', 
#         'Crop storage_guidelines_are': 'Store in a cool and dry place', 
#         'Crop has_land_preperation_requirements LandPreparation instructions': 'Ploughing, Harrowing, and Leveling', 
#         'Crop has_land_preperation_requirements LandPreparation with_soil_moisture': 'Medium'}, 
#     {'cropCult:Fertilization nutrient_name': 'Potassium', 
#         'cropCult:Fertilization quantity': 'Improves crop health', 
#         'cropCult:Fertilization stage_of_application': 'Provides resistance against insect-pests'}, 
#     {'cropCult:Crop name': 'Soybean', 
#         'cropCult:Crop has_types cropCult:CropType name': 'Soybean', 
#         'cropCult:Crop has_types cropCult:CropType used_for': 'Crop Cultivation'}, 
#     {'Crop name': 'Soybean', 
#         'Crop needs_pest_treatements PestTreatements pest_name': 'Gram pod borer and Tobacco Caterpillar', 
#         'Crop needs_pest_treatements PestTreatements pest_symptoms': 'Heavy yield losses'}, 
#     {'cropCult:Crop name': 'Soybean', 
#         'cropCult:Crop needs_seed_sowing_spacing cropCult:SeedSpacing sowing_time': 'The zone-wise details of recommended date of sowing, seed rate and spacing is given in table 3 given below.'}]
