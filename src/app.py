import streamlit as st
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient, models

# Initialize Qdrant client (local)
client = QdrantClient(":memory:")

# Initialize Ollama
llm = Ollama(model="phi3")

# Initialize Ollama embedding model
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Set up ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


# Function to load and index documents
def load_and_index_documents(directory, existing_index=None):
    documents = SimpleDirectoryReader(directory).load_data()
    vector_store = QdrantVectorStore(client=client, collection_name="documents")

    if existing_index:
        # Update existing index with new documents
        existing_index.insert_nodes(documents)
        return existing_index
    else:
        # Create new index
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context, vector_store=vector_store
        )
        return index


# Streamlit app
st.title("Document Q&A with LlamaIndex, Qdrant, and Ollama")

# Initialize session state for storing the index and uploaded files
if "index" not in st.session_state:
    st.session_state.index = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents", accept_multiple_files=True, type=["txt", "pdf"]
)

if uploaded_files:
    # Create a temporary directory to store uploaded files
    docs_dir = "documents"
    os.makedirs(docs_dir, exist_ok=True)

    new_files = False
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            new_files = True
            st.session_state.uploaded_files.add(file.name)
            with open(os.path.join(docs_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

    if new_files:
        # Load and index documents
        with st.spinner("Indexing new documents..."):
            st.session_state.index = load_and_index_documents(
                docs_dir, st.session_state.index
            )
        st.success("Documents indexed successfully!")

    # Query input
    query = st.text_input("Ask a question about the uploaded documents:")

    if query and st.session_state.index:
        with st.spinner("Generating answer..."):
            query_engine = st.session_state.index.as_query_engine()
            response = query_engine.query(query)
            st.write("Answer:", response.response)

# Cleanup: Remove temporary directory and its contents
if os.path.exists("documents"):
    for file in os.listdir("documents"):
        os.remove(os.path.join("documents", file))
    os.rmdir("documents")
