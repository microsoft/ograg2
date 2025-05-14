# Ontology Generated Retrieval Augmented Generation (OG-RAG)
![OG-RAG: Ontology-Grounded Retrieval-Augmented Generation](https://arxiv.org/html/2412.15235v1/x1.png)

**OG-RAG** enhances Large Language Models (LLMs) with domain-specific ontologies for improved factual accuracy and contextually relevant responses in fields with specialized workflows like agriculture, healthcare, knowledge work, and more.

[**Paper:** OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models](https://arxiv.org/html/2412.15235v1)

---

## 🔍 Overview
![OG-RAG Flow](https://arxiv.org/html/2412.15235v1/x2.png)

OG-RAG addresses traditional Retrieval-Augmented Generation (RAG) limitations by using hypergraphs to incorporate ontology-grounded knowledge. It retrieves minimal, highly relevant contexts, significantly boosting response accuracy and factual grounding.

---

## 📈 Key Features

* **Ontology-Grounded Retrieval**
* **Hypergraph Context Representation**
* **Optimized Context Retrieval Algorithm**
* **Enhanced Factual Accuracy**

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/og-rag.git
cd og-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a YAML config file with your environment and preferences:

```yaml
model:
  api_base: <API_BASE>
  api_key: <API_KEY>
  deployment_name: <LLM model name, eg. "gpt-4-turbo">
  api_type: <Eg. "openai">
  api_version: <Eg '2024-08-06'>

embedding_model:
  api_base: <API_BASE>
  api_key: <API_KEY>
  deployment_name: <LLM embedding model name, eg. "text-embedding-ada-002">
  api_type: <Eg. azure>
  api_version: <Eg '2024-08-06'>

data:
  documents_dir: data/md/soybean
  ontology_path: data/ontology/farm_cropcultivation_schema_ontology_jsonld.json
  kg_storage_path: data/kg/soybean
  index_dir: index_openai/vector_soybean
  subdir: False
  smart_pdf: True
  chunk_size: 8192

query:
  framework: ontohypergraph-rag
  batch_size: 10
  mode: json
  questions_file:

question_generator:
  framework: ontodocragas
  test_size: 100
  distr:
    simple: 0
    reasoning: 1
    multi_context: 0

evaluator:
  eval_file:
  reference_free: True
  type: single
  metrics:
    - answer_correctness
    - faithfulness
    - answer_similarity
    - answer_relevancy
    - context_relevancy
    - context_precision
    - context_recall
    - context_entity_recall
```

## 🚀 Usage

### Mapping Ontology and Generating Knowledge Graph

Map ontology only and Generate full knowledge graph (triples):

```bash
python build_knowledge_graph.py --config_file <path-to-config-file>
```

### Querying LLM

Execute queries:

```bash
python query_llm.py --config_file <path-to-config-file>
```

### Testing

Run tests and evaluate model performance:

```bash
python test_answers.py --config_file <path-to-config-file>
```

---

## 📚 Reference

* [**Paper:** OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models](https://arxiv.org/html/2412.15235v1)



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