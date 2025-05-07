import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any

from azureml.rag.utils.connections import (
    get_connection_by_id_v2,
    get_metadata_from_connection,
    get_target_from_connection,
    connection_to_credential,
)

from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import download_loader, ServiceContext
# from llama_index.legacy.embeddings import LangchainEmbedding
from llama_index.core.schema import Document

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import (
    GPTVectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.settings import _Settings
# from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import json
import numpy as np
import itertools
from collections import defaultdict
from langchain_together import ChatTogether, TogetherEmbeddings, Together

MAX_TOKENS = 4096 # 12288 4096
OPENAI_CONFIG = '/workspace/farmvibes-llm/agkgcopilot/openai.yaml'
LLMSHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

def read_markdown_files(file_paths: List[Path], langchain: bool=False) -> List[Document]:
    try: 
        from llama_index.readers.file import MarkdownReader
    except:
        MarkdownReader = download_loader("MarkdownReader")
    markdownreader = MarkdownReader()

    documents = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            markdown_files = glob.glob(os.path.join(file_path, "**/*.md"), recursive=True)
        else:
            markdown_files = [file_path]

        for markdown_file_path in markdown_files:
            if not os.path.exists(markdown_file_path):
                print(f"File not found: {markdown_file_path}")
                continue  # Skip this file if it does not exist
            if langchain:
                from langchain_community.document_loaders import TextLoader
                langchain_docs = TextLoader(markdown_file_path).load()
                documents.extend([Document.from_langchain_format(doc) for doc in langchain_docs])
            else:
                documents.extend(markdownreader.load_data(markdown_file_path))

    return documents

def read_pdf_files(file_paths: List[Path], smart=True) -> List[Document]:
    if smart:
        try: 
            from llama_index.readers.smart_pdf_loader import SmartPDFLoader
        except:
            os.system('pip install llama-index-readers-smart-pdf-loader')
            from llama_index.readers.smart_pdf_loader import SmartPDFLoader
        pdfreader = SmartPDFLoader(llmsherpa_api_url=LLMSHERPA_API_URL)
    else:
        try: 
            from llama_index.readers.file import PDFReader
        except:
            os.system('pip install llama-index-reader-pdf-loader')
            from llama_index.readers.file import PDFReader
        pdfreader = PDFReader(return_full_document=True)

    documents = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            pdf_files = glob.glob(os.path.join(file_path, "**/*.pdf"), recursive=True)
        else:
            pdf_files = [file_path]

        for pdf_file_path in pdf_files:
            if not os.path.exists(pdf_file_path):
                print(f"File not found: {pdf_file_path}")
                continue  # Skip this file if it does not exist
            try:
                documents.extend(pdfreader.load_data(str(pdf_file_path)))
            except:
                pass

    return documents

def load_llm_and_embeds(model_config: Dict[str, Any], embedding_config: Dict[str, Any]) -> Tuple[ChatOpenAI, Embeddings]:
    try:
        subscription_id, resource_group, workspace = get_workspace_info()
        model_config = get_openai_connection(subscription_id, resource_group, workspace, model_config['connection_id'])
    except:
        pass

    api_type = model_config.get("api_type", "azure")
    api_version = model_config.get("api_version", "2024-02-15-preview")
    resource_endpoint = model_config.get("api_base")
    api_key = model_config.get("api_key")
    deployment_name = model_config.get('deployment_name') #, deployment_name)

    if api_type in ['azure', 'openai']:
        if not (resource_endpoint and api_key):
            llm = ChatOpenAI(
                api_key=api_key,
                model=deployment_name,
                temperature=0.0,
                max_retries=15,
                max_tokens=MAX_TOKENS,
            )
            
            # raise ValueError("Required connection details are missing.")
        else:
            llm = AzureChatOpenAI(
                azure_endpoint=resource_endpoint,
                openai_api_version=api_version,
                openai_api_key=api_key,
                openai_api_type=api_type,
                deployment_name=deployment_name,
                temperature=0.0,
                max_retries=15,
                max_tokens=MAX_TOKENS,
            )
    elif api_type == 'llama':
        llm = ChatTogether(
                model=deployment_name,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
        )

    if embedding_config['api_type'] == 'azure':
        embedding_llm = AzureOpenAIEmbeddings(
            azure_endpoint=embedding_config.api_base,
            openai_api_version=embedding_config.api_version,
            openai_api_key=embedding_config.api_key,
            model=embedding_config.deployment_name,
            check_embedding_ctx_length=False,
            chunk_size=1000,
        )
    elif embedding_config['api_type'] == 'openai':
        embedding_llm = OpenAIEmbeddings(
            openai_api_key=embedding_config.api_key,
            model=embedding_config.deployment_name,
            check_embedding_ctx_length=False,
            chunk_size=1000,
        )
    else:
        # community embeddings
        embedding_llm = HuggingFaceEmbeddings(model_name=embedding_config.deployment_name)


    return llm, embedding_llm


def create_service_context(model_config: Dict[str, Any], embedding_config: Dict[str, Any]) -> ServiceContext:
    try:
        subscription_id, resource_group, workspace = get_workspace_info()
        model_config = get_openai_connection(subscription_id, resource_group, workspace, model_config['connection_id'])
    except:
        pass

    api_type = model_config.get("api_type", "azure")
    api_version = model_config.get("api_version", "2024-02-15-preview")
    resource_endpoint = model_config.get("api_base")
    api_key = model_config.get("api_key")
    deployment_name = model_config.get('deployment_name') #, deployment_name)

    if api_type in ['azure', 'openai']:
        if not (resource_endpoint and api_key):
            llm = ChatOpenAI(
                api_key=api_key,
                model=deployment_name,
                temperature=0.0,
                max_retries=15,
                max_tokens=MAX_TOKENS,
            )
            
            # raise ValueError("Required connection details are missing.")
        else:
            llm = AzureChatOpenAI(
                azure_endpoint=resource_endpoint,
                openai_api_version=api_version,
                openai_api_key=api_key,
                openai_api_type=api_type,
                deployment_name=deployment_name,
                temperature=0.0,
                max_retries=15,
                max_tokens=MAX_TOKENS,
            )
    elif api_type == 'llama':
        llm = ChatTogether(
                model=deployment_name,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
        )

    if embedding_config['api_type'] == 'azure':
        embedding_llm = AzureOpenAIEmbeddings(
            azure_endpoint=embedding_config.api_base,
            openai_api_version=embedding_config.api_version,
            openai_api_key=embedding_config.api_key,
            model=embedding_config.deployment_name,
            check_embedding_ctx_length=False,
            chunk_size=1000,
        )
    elif embedding_config['api_type'] == 'openai':
        embedding_llm = OpenAIEmbeddings(
            openai_api_key=embedding_config.api_key,
            model=embedding_config.deployment_name,
            check_embedding_ctx_length=False,
            chunk_size=1000,
        )
    else:
        # community embeddings
        embedding_llm = HuggingFaceEmbeddings(model_name=embedding_config.deployment_name)

    from llama_index.embeddings.langchain import LangchainEmbedding
    lc_embeddings = LangchainEmbedding(embedding_llm)

    service_context = _Settings(
        _llm=llm,
        _embed_model=lc_embeddings,
    )
    #     chunk_size=8192,
    #     num_output=8192,
    #     context_window=12288,
    #     chunk_overlap=128,
    # )

    return service_context


def get_workspace_info():
    """
    Retrieve the workspace information from the MLFLOW_TRACKING_URI environment variable.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"]
    uri_segments = uri.split("/")
    subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
    resource_group_name = uri_segments[uri_segments.index("resourceGroups") + 1]
    workspace_name = uri_segments[uri_segments.index("workspaces") + 1]
    return subscription_id, resource_group_name, workspace_name


def get_openai_connection(subscription_id: str, resource_group: str, workspace: str, connection_id: str):
    """
    Retrieve the OpenAI connection information.
    """
    connection_string = (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/"
        f"providers/Microsoft.MachineLearningServices/workspaces/{workspace}/"
        f"connections/{connection_id}"
    )
    connection = get_connection_by_id_v2(connection_string)
    metadata = get_metadata_from_connection(connection)
    target = get_target_from_connection(connection)
    credential = connection_to_credential(connection)

    # Catch default of '_' as target
    if len(target) <= 1:
        target = None

    # if OAI type, it could have a base url
    if len(metadata.get("base_url", "")) > 0:
        target = metadata["base_url"]

    model_config = {
        "api_base": target,
        "api_key": credential.key,  # type: ignore
        "api_version": "2023-05-15",
        "api_type": metadata.get("ApiType", "openai"),
    }

    # Add organization if it exists
    if model_config["api_type"] == "openai" and "organization" in metadata and len(metadata["organization"]) > 0:
        model_config["organization"] = metadata["organization"]

    print("Connection details:", model_config)
    return model_config


def get_documents(input_dir: str, subdir: bool=False, smart_pdf: bool=True, full_text: bool=True) -> List[Document]:
    subdirectories = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    if not subdirectories or not subdir:
        subdirectories = [Path(input_dir)]

    for subdir in subdirectories:
        print(f"Processing subdirectory: {subdir}")

        markdown_paths = list(subdir.glob("*.md"))
        documents = read_markdown_files(markdown_paths, langchain=full_text)
        pdf_paths = list(subdir.glob("*.pdf"))
        documents += read_pdf_files(pdf_paths, smart=smart_pdf and not full_text)
        
    return documents

def create_or_load_index(index_directory: str, service_context: ServiceContext, documents: Optional[Sequence[Document]] = None) -> VectorStoreIndex:
    try:
        vector_storage_context = StorageContext.from_defaults(persist_dir=index_directory)
        index = load_index_from_storage(vector_storage_context, service_context=service_context)
    except (ValueError, FileNotFoundError):
        assert documents is not None
        print (f"Index not found in folder {index_directory}. Creating a new index")
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        index.storage_context.persist(index_directory)
    assert type(index) is VectorStoreIndex
    return index


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

def load_graph_nodes_chunks(ontology_nodes_path: str, chunks: List[Document]):
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


def cosine_similarity (x, y):
    if type(y[0]) is list:
        sims = []
        for yi in y: sims.append(cosine_similarity(x, yi))
        return sims.max()
    return np.dot(np.array(x), np.array(y))/(np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(y)))

def flatten_tree (node: Dict[str, Any]):
    node_type = ''
    for k, v in node.items():
        if '@type' in k:
            node_type = v.split(':')[1] if ':' in v else v
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

def flatten_tree_sep (node: Dict[str, Any]):
    node_type = ''
    for k, v in node.items():
        if '@type' in k:
            node_type = v
            break
    node_contexts = defaultdict(lambda: set())
    for k, v in node.items():
        k_type = f'{node_type} {k}' if node_type != '' else k
        if '@type' in k:
            continue
        elif isinstance(v, dict):
            for v_node in flatten_tree(v):
                for k2, v2 in v_node.items(): node_contexts[f'{k_type} {k2}'].add(v2)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for v_node in flatten_tree(item):
                        for k2, v2 in v_node.items(): node_contexts[f'{k_type} {k2}'].add(v2)
                else:
                    node_contexts[f'{k_type}'].add(item)
        else:
            node_contexts[f'{k_type}'].add(v)
    # print ({ki: len(vi) for ki, vi in node_contexts.items()})
    node_values = itertools.product(*node_contexts.values())
    flattened_nodes = []
    for node_value in node_values:
        flattened_nodes.append({k: v for k, v in zip(node_contexts.keys(), node_value)})
    return flattened_nodes

def flatten_tree_single (node: Dict[str, Any]):
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
            for k2, v2 in flatten_tree_single(v).items():
                flattened_node[f'{node_type} {k} {k2}'] = v2
        else:
            flattened_node[f'{node_type} {k}'] = v
    return flattened_node
