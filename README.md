# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


# Config

```yaml
model:
  api_base: \<API_BASE\>
  api_key: \<API_KEY\>
  deployment_name: gpt-4-turbo
  api_type: openai
  # api_version: 2023-08-01-preview
  api_version: '2024-08-06'
  # MODEL_NAME: gpt-4-32k

embedding_model:
  api_base: \<API_BASE\>
  api_key: \<API_KEY\>
  deployment_name: text-embedding-ada-002 # sentence-transformers/all-mpnet-base-v2
  api_type: azure #huggingface
  api_version: '2024-08-06'

data:
  documents_dir: data/md/soybean # Path to the documents files
  ontology_path: data/ontology/farm_cropcultivation_schema_ontology_jsonld.json # ontology path
  kg_storage_path: data/kg/soybean # path where the triples are stored
  index_dir: index_openai/vector_soybean # path where the vector index is stored
  subdir: False
  smart_pdf: True
  chunk_size: 8192
  
query:
  framework: ontohypergraph-rag
  # framework: llm # = LLM as above
  # framework: rag # = traditional RAG 
  # framework: graphrag # = GRAPH RAG
  # framework: raptor-rag # = RAPTOR RAG
  # ontohypergraph-rag # OG RAG
  # kg-rag # Previous KG RAG
  hyperparams:
    # add hyperparams here
  batch_size: 10
  mode: json
  questions_file: # questions file

question_generator:
  framework: ontodocragas
  test_size: 100
  distr: 
    simple: 0
    reasoning: 1
    multi_context: 0

evaluator:
  eval_file: # file with questions, answers, and optionally contexts to evaluate 
  reference_free: True
  type: single
  metrics:
    # metrics to evaluate, following are RAGAS metrics
    - answer_correctness
    - faithfulness
    - answer_similarity
    - answer_relevancy
    - context_relevancy
    - context_precision
    - context_recall
    - context_entity_recall
```

You can also change any of these options using command line arguments by simply writing `--model.development_name` and so on.

# Mapping Ontology

> python build_knowledge_graph.py --config_file \<path-to-config-file\> --only_map_ontology

If you also want to generate triples, don't add `--only_map_ontology` option. 

# Querying LLM

> python query_llm.py --config_file \<path-to-config-file\>

# Testing

> python test_answers.py --config_file \<path-to-config_file\>