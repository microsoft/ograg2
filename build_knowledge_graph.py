from concurrent.futures import ThreadPoolExecutor
from knowledge_graph import create_kg_triples, KGGenerator
from ontology_mapping import OntologyMapping
from utils import create_service_context, get_documents, create_or_load_index, get_config, load_llm_and_embeds
import os
import time

if __name__ == '__main__':
    config = get_config()
    
    # service_context = create_service_context(config.model, config.embedding_model) 
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)
    documents = get_documents(config.data.documents_dir, 
                              subdir=config.data.subdir, 
                              smart_pdf=config.data.smart_pdf,
                              full_text=config.data.full_text,)

    to_map_ontology = False
    to_create_kg = False

    if os.path.isdir(config.data.kg_storage_path):
        graph_files = [x for x in os.listdir(config.data.kg_storage_path) if 'pkl' in x]
        if len(graph_files) == 0:
            to_create_kg = True
            ontology_dir = [x for x in os.listdir(config.data.kg_storage_path) if 'ontology' in x]
            if len(ontology_dir) == 0:
                to_map_ontology = True
    else:
        to_create_kg = True
        to_map_ontology = True

    start_time = time.time()
    if to_map_ontology or config.options.force_map_ontology:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            ontology_creator = OntologyMapping(
                ontology_context_definition_path=config.data.ontology_path,
                llm=llm,
                documents=documents,
                chunk_size=config.data.chunk_size if 'chunk_size' in config.data else 8192,
            )
            current_output_dir = f'{config.data.kg_storage_path}/ontology'
            os.makedirs(current_output_dir, exist_ok=True)
            ontology_creator.generate_and_save_ontology_data(executor, current_output_dir)
    print (config.data.ontology_path, time.time() - start_time)
    
    if (to_create_kg or config.options.force_create_kg_triples) and not config.options.only_map_ontology:
        create_kg_triples(
            input_directory=config.data.kg_storage_path,
            output_directory=config.data.kg_storage_path,
            llm=llm,
            batch_size=config.query.batch_size,
        )
    