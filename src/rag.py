import time
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

# Initialize Qdrant client (local)
client = QdrantClient(":memory:")

# Initialize Ollama
llm = Ollama(model="phi3")

# Initialize Ollama embedding model
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Set up ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


def load_and_index_documents(directory, existing_index=None):
    start_time = time.time()
    documents = SimpleDirectoryReader(directory).load_data()
    vector_store = QdrantVectorStore(client=client, collection_name="documents")

    if existing_index:
        # Update existing index with new documents
        existing_index.insert_nodes(documents)
        index = existing_index
    else:
        # Create new index
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context, vector_store=vector_store
        )

    end_time = time.time()
    indexing_time = end_time - start_time
    return index, indexing_time


def query_index(index, query):
    query_start_time = time.time()
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    query_end_time = time.time()
    query_time = query_end_time - query_start_time
    return response.response, query_time


def cleanup_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        os.rmdir(directory)
