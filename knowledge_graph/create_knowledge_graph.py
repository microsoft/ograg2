from pathlib import Path
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import ast

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import (BaseNode, Document)
from llama_index.core.prompts.base import PromptTemplate, BasePromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import (KnowledgeGraphIndex, ServiceContext, StorageContext)


DEFAULT_KG_TRIPLET_ONTOLOGY_EXTRACT_TMPL = """
Using the @graph namespace in the following json-ld, generate a complete python list of tuples of triples for knowledge graph in the format (subject, predicate, object). 
Keep the property names exactly as it is in the Json-ld.
The 'subject', 'predicate', and 'object' can only be strings. 
Subjects and objects should be in natural language.
Make sure that the predicate is structured so that it is a grammatically correct phrase.
The triples cannot be nested, so please flatten them. Also do not include triples keys of "subject", "object", "predicate", only the values. 
For nested structure, within "@graph" object, such as "xy": \{\{"k": "v" }} flatten it by rearranging keys "xy", "k" to either "xyk", "xky", or "kxy" in a way that it grammatically makes sense. 
Generate all triples.
Do not add any other text in response other than the list of tuples of triples.
------------------------------

JSON-LD:
{data}

"""

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
"""

KG_TRIPLET_ONTOLOGY_EXTRACT_PROMPT = PromptTemplate(
    KG_TRIPLET_ONTOLOGY_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)



ONTOLOGY_JSONLD_DATA_CREATE_TMPL = """
Here is a context definition for wheat crop cultivation ontology.

Context Definition: 

{context_definition}

-----------------

Generate a JSON-LD using the following data and the above context definition for crop cultivation ontology.
Use '@graph' object namespace for the data in JSON-LD.
Be comprehensive and make sure to Fill all of the data. 
Keep nesting to the minimum and still be able to disambiguate.
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
"""

ONTOLOGY_JSONLD_DATA_CREATE_PROMPT = PromptTemplate(
    ONTOLOGY_JSONLD_DATA_CREATE_TMPL, prompt_type="structured_fill"
)

class KnowledgeGraphOntologyIndex():
    """Knowledge Graph Index.

    Build a KG by extracting triplets from Ontologies
    
    Args:
    service_context: Needs a high num_output (tested with 512), high chunk_size (tested with 8192) and context window (tested with 8192)
    """
    
    
    def __init__(
        self,
        ontology_context_definition_path: str = None,
        documents: Optional[Sequence[Document]] = None,
        pdf_document_path: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        storage_path: Optional[str] = None,
        graph_storage_context: Optional[StorageContext] = None,
        ontology_jsonld_data_create_prompt: Optional[BasePromptTemplate] = None,
        kg_triple_generation_prompt: Optional[BasePromptTemplate] = None,
        include_embeddings: bool = False,
        verbose: bool = False,
        show_embedding_progress: bool = False,
        continuous_save: bool = False,
        debug: bool=False,
        **kwargs: Any
    ) -> None:
        
        """Initialize params."""
        # need to set parameters before building index in base class.
        
        self.ontology_jsonld_data_create_prompt = (
            ontology_jsonld_data_create_prompt or ONTOLOGY_JSONLD_DATA_CREATE_PROMPT
        )
        
        self.kg_triple_generation_prompt = (
            kg_triple_generation_prompt or KG_TRIPLET_ONTOLOGY_EXTRACT_PROMPT
        )
        
        
    
        self.service_context=service_context
        
        self.storage_path=None
        if storage_path is not None:
            print("If you are using graph store, use graph_storage_context")
            self.storage_path = storage_path
        
        self.graph_storage_context=None
        if graph_storage_context is not None:
            assert (
                graph_storage_context.graph_store is not None
            ), "Must provide a graph store in the storage context."
            assert(storage_path is None), "Must only provide one storage context"
            self.graph_storage_context = graph_storage_context
        
        
        self.documents = ( documents or self._from_pdf(pdf_document_path) )
        self._nodes = self.get_nodes()
        
        self._ontology_g0 = self._load_context_definition(ontology_context_definition_path)
        
        self._verbose = verbose
        
        self._debug=debug
        
        self._nodes_ontologies_g2 = None
        
        self._kg_all_triples = None
        
        self._fmt_nodes_g2create_prompt = None
        
        self._fmt_nodes_kgcreate_prompt = None
        
        self._nodes_kg_triples = self.generate_and_save_node_triples() if continuous_save is True else self.generate_node_triples()
        
            

    def _load_context_definition(self, json_file_path: str):
        with open(json_file_path) as f:
            return f.read()
            
    def _from_pdf(self, pdf_file_path: str) -> List[Document]:
        """For PDF files only with text. No tables or figures"""
        
        reader = PDFReader()
        pdf_path = Path(pdf_file_path)
        return reader.load_data(file=pdf_path)
    
    def get_nodes(self) -> List[BaseNode]:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=8192)
        return node_parser.get_nodes_from_documents(self.documents)

    def generate_node_triples(self) -> List[str]:
        nodes_ontologies_g2 = []
        nodes_kg_triples = []
        fmt_nodes_g2create_prompt = []
        fmt_nodes_kgcreate_prompt = []
        llm = self.service_context.llm
        
        for idx, node in enumerate(self._nodes):
            if self._verbose: print("Processing Node {idx}.\n Generating data for G0 Ontology:".format(idx=idx))
            
            data = node.text
            curr_fmt_nodes_g2create_prompt=self.ontology_jsonld_data_create_prompt.format(data=data,context_definition=self._ontology_g0)
            curr_ontology_g2 = llm.complete(
                curr_fmt_nodes_g2create_prompt
            ).text
            fmt_nodes_g2create_prompt.append(curr_fmt_nodes_g2create_prompt)
            
            nodes_ontologies_g2.append(curr_ontology_g2)
            
            
            if self._verbose: print("Generating Triples:")
            curr_fmt_nodes_kgcreate_prompt = self.kg_triple_generation_prompt.format(data=curr_ontology_g2)
            fmt_nodes_kgcreate_prompt.append(curr_fmt_nodes_kgcreate_prompt)
            kg_triples = llm.complete(
                curr_fmt_nodes_kgcreate_prompt
            ).text
            
            nodes_kg_triples.append(kg_triples)
        
        self._nodes_ontologies_g2 = nodes_ontologies_g2
        self._fmt_nodes_g2create_prompt = fmt_nodes_g2create_prompt
        self._fmt_nodes_kgcreate_prompt = fmt_nodes_kgcreate_prompt
        return nodes_kg_triples
    
    def get_nodes_ontologies_g2(self):
        return self._nodes_ontologies_g2
    
    def save_triples(self):
        if self.graph_storage_context is not None:
            new_index = KnowledgeGraphIndex(
                [],
                service_context=self.service_context,
                storage_context=self.graph_storage_context
            )

            nodes_kg_triples = self._nodes_kg_triples
            for triples_str,node in zip(nodes_kg_triples,self._nodes):
                triples = ast.literal_eval(triples_str)
                for tup in triples:
                    try:
                        new_index.upsert_triplet_and_node(tup, node)
                    except TypeError:
                        if self._verbose is True: print("Error storing following triple in the graph store: " + tup)
        elif self.storage_path is not None:
            for triples_str,node in zip(nodes_kg_triples,self._nodes):
                Path(self.storage_path).mkdir(parents=True, exist_ok=True)
                with open(self.storage_path + "/" + node.node_id, 'w') as file:
                    file.write(triples_str)
        else:
            raise Exception("Must provide a graph storage context or storage path if no KG triplets provided.")
            
                        
    def get_all_triples(self):
        nodes_kg_triples = self._nodes_kg_triples
        triples = []
        
        for triples_str in nodes_kg_triples:
            triples = triples + ast.literal_eval(triples_str)
        
        self._kg_all_triples = triples
        
        return triples
    
    def generate_and_save_node_triples(self) -> List[str]:
        nodes_ontologies_g2 = []
        nodes_kg_triples = []
        fmt_nodes_g2create_prompt = []
        fmt_nodes_kgcreate_prompt = []
        
        if self.graph_storage_context is not None:
            new_index = KnowledgeGraphIndex(
                [],
                service_context=self.service_context,
                storage_context=self.storage_context
            )
            
            for idx, node in enumerate(self._nodes):
                if self._verbose: print("Processing Node {idx}.\n Generating data for G0 Ontology:".format(idx=idx))
                llm = self.service_context.llm
                
                data = node.text
                curr_fmt_nodes_g2create_prompt=self.ontology_jsonld_data_create_prompt.format(data=data,context_definition=self._ontology_g0)
                curr_ontology_g2 = llm.complete(
                    curr_fmt_nodes_g2create_prompt
                ).text
                fmt_nodes_g2create_prompt.append(curr_fmt_nodes_g2create_prompt)
                
                nodes_ontologies_g2.append(curr_ontology_g2)
                
                
                if self._verbose: print("Generating Triples:")
                curr_fmt_nodes_kgcreate_prompt = self.kg_triple_generation_prompt.format(data=curr_ontology_g2)
                fmt_nodes_kgcreate_prompt.append(curr_fmt_nodes_kgcreate_prompt)
                kg_triples = llm.complete(
                    curr_fmt_nodes_kgcreate_prompt
                ).text
                
                kg_triples_list = ast.literal_eval(kg_triples)
                if self._verbose: print("Saving Triples:")
                for tup in kg_triples_list:
                    try:
                        new_index.upsert_triplet_and_node(tup, node)
                    except TypeError:
                        if self._verbose is True: print("Error storing following triple in the graph store: " + tup)
                
                nodes_kg_triples.append(kg_triples)
        
        elif self.storage_path is not None:
            for idx, node in enumerate(self._nodes):
                if self._verbose: print("Processing Node {idx}.\n Generating data for G0 Ontology:".format(idx=idx))
                llm = self.service_context.llm
                
                data = node.text
                curr_fmt_nodes_g2create_prompt=self.ontology_jsonld_data_create_prompt.format(data=data,context_definition=self._ontology_g0)
                if self._debug is True:
                    print("""###############################
                        -------------------------------""") 
                    print("G0 Ontology generation prompt = " + curr_fmt_nodes_g2create_prompt)
                    stream = llm.stream_complete(curr_fmt_nodes_g2create_prompt)
                    for chunk in stream:
                        print(chunk.delta, end="")
                    
                    curr_ontology_g2 = chunk.text
                else:        
                    curr_ontology_g2 = llm.complete(
                        curr_fmt_nodes_g2create_prompt
                    ).text
                
                    
                fmt_nodes_g2create_prompt.append(curr_fmt_nodes_g2create_prompt)
                
                nodes_ontologies_g2.append(curr_ontology_g2)
                
                
                if self._verbose: print("Generating Triples:")
                curr_fmt_nodes_kgcreate_prompt = self.kg_triple_generation_prompt.format(data=curr_ontology_g2)
                fmt_nodes_kgcreate_prompt.append(curr_fmt_nodes_kgcreate_prompt)
                if self._debug is True: print("Triple Generation Prompt = " + curr_fmt_nodes_kgcreate_prompt)
                kg_triples = llm.complete(
                    curr_fmt_nodes_kgcreate_prompt
                ).text
            
                if self._verbose: print("Saving Triples:")
                Path(self.storage_path).mkdir(parents=True, exist_ok=True)
                with open(self.storage_path + "/" + node.node_id, 'w') as file:
                    file.write(kg_triples)
                
                nodes_kg_triples.append(kg_triples)
        
        self._nodes_ontologies_g2 = nodes_ontologies_g2
        self._fmt_nodes_g2create_prompt = fmt_nodes_g2create_prompt
        self._fmt_nodes_kgcreate_prompt = fmt_nodes_kgcreate_prompt
        return nodes_kg_triples
    

class KnowledgeGraphOntologyIndex_Deprecated():
    """Knowledge Graph Index.

    Build a KG by extracting triplets from Ontologies
    
    Args:
    service_context: Needs a high num_output (tested with 512), high chunk_size (tested with 8192) and context window (tested with 8192)
    """
    
    
    def __init__(
        self,
        ontology_context_definition_path: str = None,
        documents: Optional[Sequence[Document]] = None,
        pdf_document_path: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        ontology_jsonld_data_create_prompt: Optional[BasePromptTemplate] = None,
        kg_triple_generation_prompt: Optional[BasePromptTemplate] = None,
        include_embeddings: bool = False,
        verbose: bool = False,
        show_embedding_progress: bool = False,
        continuous_save: bool = False,
        **kwargs: Any,
    ) -> None:
        
        """Initialize params."""
        # need to set parameters before building index in base class.
        
        self.ontology_jsonld_data_create_prompt = (
            ontology_jsonld_data_create_prompt or ONTOLOGY_JSONLD_DATA_CREATE_PROMPT
        )
        
        self.kg_triple_generation_prompt = (
            kg_triple_generation_prompt or KG_TRIPLET_ONTOLOGY_EXTRACT_PROMPT
        )
        
        
    
        self.service_context=service_context
        self.storage_context=None
        if storage_context is not None:
            assert (
                storage_context.graph_store is not None
            ), "Must provide a graph store in the storage context."
            self.storage_context = storage_context
        
        
        self.documents = ( documents or self._from_pdf(pdf_document_path) )
        self._nodes = self.get_nodes()
        
        self._ontology_g0 = self._load_context_definition(ontology_context_definition_path)
        
        self._verbose = verbose
        
        self._nodes_ontologies_g2 = None
        
        self._kg_all_triples = None
        
        self._fmt_nodes_g2create_prompt = None
        
        self._fmt_nodes_kgcreate_prompt = None
        
        self._nodes_kg_triples = self.generate_and_save_node_triples() if continuous_save is True else self.generate_node_triples()
        
            

    def _load_context_definition(self, json_file_path: str):
        with open(json_file_path) as f:
            return f.read()
            
    def _from_pdf(self, pdf_file_path: str) -> List[Document]:
        """For PDF files only with text. No tables or figures"""
        
        reader = PDFReader()
        pdf_path = Path(pdf_file_path)
        return reader.load_data(file=pdf_path)
    
    def get_nodes(self) -> List[BaseNode]:
        node_parser = SimpleNodeParser.from_defaults(chunk_size=8192)
        return node_parser.get_nodes_from_documents(self.documents)

    def generate_node_triples(self) -> List[str]:
        nodes_ontologies_g2 = []
        nodes_kg_triples = []
        fmt_nodes_g2create_prompt = []
        fmt_nodes_kgcreate_prompt = []
        llm = self.service_context.llm
        
        for node in self._nodes:
            if self._verbose: print("Generating data for G0 Ontology:")
            
            data = node.text
            curr_fmt_nodes_g2create_prompt=self.ontology_jsonld_data_create_prompt.format(data=data,context_definition=self._ontology_g0)
            curr_ontology_g2 = llm.complete(
                curr_fmt_nodes_g2create_prompt
            ).text
            fmt_nodes_g2create_prompt.append(curr_fmt_nodes_g2create_prompt)
            
            nodes_ontologies_g2.append(curr_ontology_g2)
            
            
            if self._verbose: print("Generating Triples:")
            curr_fmt_nodes_kgcreate_prompt = self.kg_triple_generation_prompt.format(data=curr_ontology_g2)
            fmt_nodes_kgcreate_prompt.append(curr_fmt_nodes_kgcreate_prompt)
            kg_triples = llm.complete(
                curr_fmt_nodes_kgcreate_prompt
            ).text
            
            nodes_kg_triples.append(kg_triples)
        
        self._nodes_ontologies_g2 = nodes_ontologies_g2
        self._fmt_nodes_g2create_prompt = fmt_nodes_g2create_prompt
        self._fmt_nodes_kgcreate_prompt = fmt_nodes_kgcreate_prompt
        return nodes_kg_triples
    
    def get_nodes_ontologies_g2(self):
        return self._nodes_ontologies_g2
    
    def save_triples(self):
        assert self.storage_context is not None, "Must provide a storage context if no KG triplets provided."
        new_index = KnowledgeGraphIndex(
            [],
            service_context=self.service_context,
            storage_context=self.storage_context
        )

        nodes_kg_triples = self._nodes_kg_triples
        for triples_str,node in zip(nodes_kg_triples,self._nodes):
            triples = ast.literal_eval(triples_str)
            for tup in triples:
                try:
                    new_index.upsert_triplet_and_node(tup, node)
                except TypeError:
                    if self._verbose is True: print("Error storing following triple in the graph store: " + tup)
                    
    def get_all_triples(self):
        nodes_kg_triples = self._nodes_kg_triples
        triples = []
        
        for triples_str in nodes_kg_triples:
            triples = triples + ast.literal_eval(triples_str)
        
        self._kg_all_triples = triples
        
        return triples
    
    def generate_and_save_node_triples(self) -> List[str]:
        nodes_ontologies_g2 = []
        nodes_kg_triples = []
        fmt_nodes_g2create_prompt = []
        fmt_nodes_kgcreate_prompt = []
        
        assert self.storage_context is not None, "Must provide a storage context if no KG triplets provided."
        new_index = KnowledgeGraphIndex(
            [],
            service_context=self.service_context,
            storage_context=self.storage_context
        )
        
        for node in self._nodes:
            if self._verbose: print("Generating data for G0 Ontology:")
            llm = self.service_context.llm
            
            data = node.text
            curr_fmt_nodes_g2create_prompt=self.ontology_jsonld_data_create_prompt.format(data=data,context_definition=self._ontology_g0)
            curr_ontology_g2 = llm.complete(
                curr_fmt_nodes_g2create_prompt
            ).text
            fmt_nodes_g2create_prompt.append(curr_fmt_nodes_g2create_prompt)
            
            nodes_ontologies_g2.append(curr_ontology_g2)
            
            
            if self._verbose: print("Generating Triples:")
            curr_fmt_nodes_kgcreate_prompt = self.kg_triple_generation_prompt.format(data=curr_ontology_g2)
            fmt_nodes_kgcreate_prompt.append(curr_fmt_nodes_kgcreate_prompt)
            kg_triples = llm.complete(
                curr_fmt_nodes_kgcreate_prompt
            ).text
            
            kg_triples_list = ast.literal_eval(kg_triples)
            if self._verbose: print("Saving Triples:")
            for tup in kg_triples_list:
                try:
                    new_index.upsert_triplet_and_node(tup, node)
                except TypeError:
                    if self._verbose is True: print("Error storing following triple in the graph store: " + tup)
            
            nodes_kg_triples.append(kg_triples)
        
        self._nodes_ontologies_g2 = nodes_ontologies_g2
        self._fmt_nodes_g2create_prompt = fmt_nodes_g2create_prompt
        self._fmt_nodes_kgcreate_prompt = fmt_nodes_kgcreate_prompt
        return nodes_kg_triples
