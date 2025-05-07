from utils import create_service_context, get_config, load_llm_and_embeds, load_graph_nodes
import os
from tqdm import tqdm

MAX_TOKENS = 1024

CONVERT_PROMPT = """Convert the following information about an entity into an english sentence. 
The information is presented as a standard jsonld ontology schema or a nested list of {{key: value}} where key is a property name and the value is its value.
Remove any redundant information but KEEP ALL the information that is important. DO NOT COMPRESS INFORMATION USING "and so on" or "etc" or "and others" etc.
---------------------
For example, 
Information: {{'name': 'John Doe', 'age': '25', 'location': 'New York'}}
Sentence: John Doe is 25 years old and lives in New York.

Information: {{'@type': 'Crop', 'name': 'Soybean',  'has_seeding_rate': {{'@type': 'schema:QuantitativeValue', 'value': '30', 'unitText': 'Kg per hectare'}}}}
Sentence: The crop soybean has 30 kg per hectare seeding rate.

---------------------

Information: {information}
---------------------
Sentence: 
"""


if __name__ == '__main__':
    config = get_config()
    service_context = create_service_context(config.model, config.embedding_model)
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)

    ontology_nodes_path = config.data.kg_storage_path

    if 'summarized_texts.txt' in os.listdir(ontology_nodes_path):
        with open(f'{ontology_nodes_path}/summarized_texts.txt', 'r') as f:
            texts = f.read().split('\n')
    else:
        nodes = load_graph_nodes(ontology_nodes_path)

        texts = []
        for node in tqdm(nodes, desc='Converting nodes to text'):
            texts.append(llm.invoke(CONVERT_PROMPT.format(information=node), max_tokens=MAX_TOKENS).content)
        with open(f'{ontology_nodes_path}/summarized_texts.txt', 'w') as f:
            f.write('\n'.join(texts))
