import argparse
import os
import glob 
import yaml
from easydict import EasyDict

HOME = '/home/ksartik/farmvibes-llm/agkgcopilot'

def get_config(no_args=False):
    if not no_args:
        parser = argparse.ArgumentParser(description="Question answering using knowledge graph augmented context.")
        parser.add_argument('--config_file', type=str, default='configs/config_soybean.yaml')
        parser.add_argument('--force_map_ontology', action='store_true')
        parser.add_argument('--only_map_ontology', action='store_true')
        parser.add_argument('--force_create_kg_triples', action='store_true')
        parser.add_argument('--force_personal_openai', action='store_true')
        parser.add_argument('--force_personal_openai_emb_only', action='store_true')
        parser.add_argument('--rewrite', action='store_true')
        parser.add_argument('--results_dir', type=str)

        args, remaining_args = parser.parse_known_args()
    else:
        args = EasyDict({'config_file': 'configs/config_soybean.yaml', 'force_map_ontology': False, 'force_create_kg_triples': False, 
                         'force_personal_openai': False, 'force_personal_openai_emb_only': True, 'results_dir': None,
                         'rewrite': False})
        remaining_args = []

    config_parser = argparse.ArgumentParser(add_help=False)
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        def add_arguments_from_config(config, root_key=''):
            for key, value in config.items():
                if type(value) is dict:
                    add_arguments_from_config(value, root_key=f'{root_key}.{key}' if root_key != '' else key)
                else:
                    config_parser.add_argument(
                        f'--{root_key}.{key}' if root_key != '' else f'--{key}', 
                        type=type(value), default=value
                    )
        add_arguments_from_config(config)


    new_args, _ = config_parser.parse_known_args(remaining_args)
    def recursive_set (cfg, keys, value): 
        if len(keys) == 1: cfg[keys[0]] = value
        else: recursive_set(cfg[keys[0]], keys[1:], value)
    for key, value in new_args.__dict__.items():
        recursive_set(config, key.split('.'), value)

    config = EasyDict(config)
    config['options'] = args.__dict__
    config.data.full_text = False if 'full_text' not in config.data else config.data.full_text
    
    if args.results_dir is not None:
        config.query.answers_file = config.query.answers_file.replace("results/", f'{args.results_dir}/')
        config.evaluator.eval_file = config.evaluator.eval_file.replace("results/", f'{args.results_dir}/')
    
    api_keys = yaml.safe_load(open('api_keys.yaml', 'r'))
    for k, v in api_keys.items():
        os.environ[k] = v
    if config.model.deployment_name.startswith('gpt'):
        if 'AZURE_API_KEY' in api_keys and not args.force_personal_openai:
            config.model.api_key = api_keys['AZURE_API_KEY']
            os.environ['OPENAI_API_KEY'] = api_keys['AZURE_API_KEY']
            os.environ['GRAPHRAG_API_KEY'] = api_keys['AZURE_API_KEY']
            os.environ['GRAPHRAG_LLM_API_KEY'] = api_keys['AZURE_API_KEY']
            config.model.api_base = api_keys['AZURE_API_BASE']
            config.model.api_version = api_keys['AZURE_API_VERSION']
            config.model.api_type = 'azure'
        else:
            config.model.api_key = api_keys['OPENAI_API_KEY']
            os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
            os.environ['GRAPHRAG_API_KEY'] = api_keys['OPENAI_API_KEY']
            os.environ['GRAPHRAG_LLM_API_KEY'] = api_keys['OPENAI_API_KEY']
    elif 'llama' in config.model.deployment_name.lower():
        os.environ['TOGETHER_API_KEY'] = api_keys['TOGETHER_API_KEY']
        config.model.api_type = 'llama'
    if config.embedding_model.deployment_name.startswith('text-embedding'):
        if 'AZURE_API_KEY' in api_keys and not args.force_personal_openai and not args.force_personal_openai_emb_only:
            config.embedding_model.api_key = api_keys['AZURE_API_KEY']
            config.embedding_model.api_type = 'azure'
            config.embedding_model.api_base = api_keys['AZURE_API_BASE']
            config.embedding_model.api_version = api_keys['AZURE_API_VERSION']
        else:
            config.embedding_model.api_key = api_keys['OPENAI_API_KEY']
            config.embedding_model.api_type = 'openai'
        
    # if config.index_dir is None:
    #     doc_name = '_'.join([os.path.basename(x)[:-3].split('_')[0] for x in glob.glob(os.path.join(config.data.documents_dir, f'**/*.md'), recursive=True)])
    #     config.index_dir = f"{HOME}/index/vector_{doc_name}"
    config.rewrite = args.rewrite
    
    return config