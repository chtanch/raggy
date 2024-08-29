from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_data(input_files):
    # reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    reader = SimpleDirectoryReader(input_files=input_files)
    docs = reader.load_data()
    return docs


def load_index(docs):
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(
        model="phi3",
        temperature=0.2,
        system_prompt="""You are an expert in extracting information from documents. 
        Keep your answers technical and based on facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


def initialize_chat_engine(index):
    return index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )


""" 
vector_store = index.vector_store


def print_embeddings_and_doc_ids(vector_store):
    # Access the embedding_dict and text_id_to_ref_doc_id
    embedding_dict = vector_store.data.embedding_dict
    text_id_to_ref_doc_id = vector_store.data.text_id_to_ref_doc_id

    # Iterate through the embedding_dict
    for text_id, embedding in embedding_dict.items():
        # Get the corresponding doc_id
        doc_id = text_id_to_ref_doc_id.get(text_id, "Unknown")

        # Print the embedding and doc_id
        st.write(f"Doc ID: {doc_id}")
        st.write(f"Embedding: {embedding}")
        st.write("-" * 50)


print_embeddings_and_doc_ids(vector_store)
"""
