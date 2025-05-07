from utils import create_service_context, get_documents, create_or_load_index, get_config
import os
from llama_index.core.retrievers import VectorIndexRetriever
from typing import Union, List
import json
from collections import defaultdict
from utils import load_llm_and_embeds
import pandas as pd
from tqdm import tqdm
import time

class QnA_IO:
    def __init__(self):
        self.data = defaultdict(lambda: [])
    
    def read(self, file_paths: Union[str, List[str]]):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            with open(file_path, 'r') as f:
                if '.json' in file_path:
                    data = json.load(f)
                elif '.csv' in file_path:
                    data = pd.read_csv(f).to_dict('records')
                for datum in data:
                    for key in datum:
                        if key != 'metadata':
                            self.data[key].append(datum[key])
    
    def write(self, file_path, **kwargs):
        # self.data['answer'] = responses
        for key in kwargs:
            self.data[key] = kwargs[key]
        if '.json' in file_path:
            new_data = [dict(zip(self.data, t)) for t in zip(*self.data.values())]
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
        elif '.csv' in file_path:
            new_data = pd.DataFrame(self.data)
            new_data.to_csv(file_path, index=False)
        else:
            raise NotImplementedError(f"Output format {self.file_type} is not implemented yet.")


if __name__ == '__main__':
    config = get_config()
    service_context = create_service_context(config.model, config.embedding_model)
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)
    
    documents = get_documents(config.data.documents_dir, subdir=config.data.subdir, smart_pdf=config.data.smart_pdf, full_text=config.data.full_text)
    
    rules = []
    if 'rules_file' in config.data and config.data.rules_file:
        with open (config.data.rules_file) as f:
            rules += [line[:-1] for line in f.readlines()]

    if config.query.method == 'kg-rag':
        from query_engine import KnowledgeTriplesGraphQueryEngine
        vector_index = create_or_load_index(index_directory=config.data.index_dir, service_context=service_context, documents=documents)
        vector_retriever = VectorIndexRetriever(index=vector_index) #, embed_model=embeddings)
        query_engine = KnowledgeTriplesGraphQueryEngine(llm=llm, embeddings=embeddings, kg_triples_storage_path=config.data.kg_storage_path, 
                                                        vector_retriever=vector_retriever, verbose=True)
    elif config.query.method == 'llm':
        from query_engine import LLMQueryEngine
        query_engine = LLMQueryEngine(llm=llm, verbose=True)
    elif config.query.method == 'rag':
        from query_engine import RAGQueryEngine
        vector_index = create_or_load_index(index_directory=config.data.index_dir, service_context=service_context, documents=documents)
        if 'hyperparams' in config.query and 'top_k' in config.query.hyperparams:
            vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=config.query.hyperparams.top_k) #, embed_model=embeddings)
        else:
            vector_retriever = VectorIndexRetriever(index=vector_index)
        query_engine = RAGQueryEngine(llm=llm, vector_retriever=vector_retriever, verbose=True)
    elif config.query.method == 'raptor-rag':
        from query_engine import RaptorQueryEngine
        from llama_index.llms.azure_openai import AzureOpenAI
        openai_llm = AzureOpenAI(
            model='gpt-4o', 
            api_key=os.environ['AZURE_API_KEY'], 
            engine='gpt-4o', 
            azure_endpoint=os.environ['AZURE_API_BASE'], 
            api_version=os.environ['AZURE_API_VERSION']
        )
        embed_model = service_context.embed_model
        query_engine = RaptorQueryEngine(documents=documents, llm=openai_llm, query_llm=llm,
                                         embed_model=embed_model, verbose=True, 
                                         **config.query.hyperparams)
        # from llama_index.packs.raptor import RaptorPack
        # from llama_index.core.base.llms.types import LLMMetadata
        # service_context.llm.metadata = LLMMetadata()
        # pack = RaptorPack(documents=documents, llm=service_context.llm, embed_model=service_context.embed_model)
    elif config.query.method == 'graphrag':
        from query_engine import GraphRAGQueryEngine
        query_engine = GraphRAGQueryEngine(llm=llm, data_config=config.data, **config.query.hyperparams)
    elif config.query.method == 'snippet-rag':
        from query_engine import SnippetRAGQueryEngine
        query_engine = SnippetRAGQueryEngine.from_ontology_path(
                            llm=llm,
                            ontology_nodes_path=f'{config.data.kg_storage_path}',
                            embed_model=embeddings
                        )
    elif config.query.method == 'ontograph-rag':
        from query_engine import OntoGraphQueryEngine
        query_engine = OntoGraphQueryEngine.from_ontology_path(
                            ontology_nodes_path=f'{config.data.kg_storage_path}',
                            llm=llm,
                            embed_model=embeddings
                        )
    elif config.query.method == 'ontohypergraph-rag':
        from query_engine import OntoHyperGraphQueryEngine
        vector_retriever = None
        if 'hyperparams' in config.query and 'vector_index' in config.query.hyperparams and config.query.hyperparams.vector_index:
            vector_index = create_or_load_index(index_directory=config.data.index_dir, service_context=service_context, documents=documents)
            vector_retriever = VectorIndexRetriever(index=vector_index)
        query_engine = OntoHyperGraphQueryEngine.from_ontology_path(
                            ontology_nodes_path=f'{config.data.kg_storage_path}',
                            llm=llm,
                            embed_model=embeddings,
                            vector_retriever=vector_retriever,
                        )
    elif config.query.method == 'fullontology-rag':
        from query_engine import FullOntoQueryEngine
        query_engine = FullOntoQueryEngine.from_ontology_path(
                            ontology_nodes_path=f'{config.data.kg_storage_path}',
                            llm=llm,
                        )
    else:
        raise NotImplementedError(f"Query method {config.query.method} is not implemented yet.")
    
    if len(config.query.questions_file) == 0:
        while True:
            query = input("Type your query (press Enter to stop):")
            if query == '':
                break
            if 'hyperparams' in config.query:
                response = query_engine.query(query_str=query, **config.query.hyperparams)
            else:
                response = query_engine.query(query_str=query)
            print (f'Response: {response}')
    else:
        answer_dir = os.path.dirname(config.query.answers_file)
        os.makedirs(answer_dir, exist_ok=True)
        if 'hyperparams' in config.query:
            answer_file = os.path.basename(config.query.answers_file)
            answer_fname, answer_ftype = answer_file.split('.')
            config.query.answers_file = f'{answer_dir}/{answer_fname}_{'_'.join([f'{k}{v}' for k, v in config.query.hyperparams.items()])}.{answer_ftype}' 
        if os.path.exists(config.query.answers_file) and not config.rewrite:
            exit(f"Answers file {config.query.answers_file} already exists.")

        evalauator = QnA_IO()
        questions = evalauator.read(config.query.questions_file)
        questions = evalauator.data["question"]
        responses = []
        times = []
        retrieved_contexts = []
        print (config.query.answers_file)
        for question in tqdm(questions):
            start_time = time.time()
            if 'hyperparams' in config.query:
                response, retrieved_context = query_engine.query(query_str=question, return_context=True, rules=rules, **config.query.hyperparams)
            else:
                response, retrieved_context = query_engine.query(query_str=question, return_context=True, rules=rules)
            try:
                responses.append(response.response)
            except:
                try:
                    responses.append(response.content)
                except:
                    responses.append(response)
            times.append(time.time() - start_time)
            retrieved_contexts.append(retrieved_context)
            print (f'Question: {question}\nResponse: {response}\nRetrieved Context: {retrieved_context}\n')
        
        evalauator.write(config.query.answers_file, answer=responses, retrieved_context=retrieved_contexts, time=times)
