import os
import streamlit as st
import time
from rag import load_and_index_documents, query_index, cleanup_directory

st.title("RAGgy")

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
    upload_start_time = time.time()
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            new_files = True
            st.session_state.uploaded_files.add(file.name)
            with open(os.path.join(docs_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
    upload_end_time = time.time()
    upload_time = upload_end_time - upload_start_time
    st.info(f"Time taken for uploading: {upload_time:.2f} seconds")

    if new_files:
        # Load and index documents
        with st.spinner("Indexing new documents..."):
            st.session_state.index, indexing_time = load_and_index_documents(
                docs_dir, st.session_state.index
            )
        st.success("Documents indexed successfully!")
        st.info(f"Time taken for indexing: {indexing_time:.2f} seconds")

    # Query input
    query = st.text_input("Ask a question about the uploaded documents:")

    if query and st.session_state.index:
        with st.spinner("Generating answer..."):
            response, query_time = query_index(st.session_state.index, query)
            st.write("Answer:", response)
            st.info(f"Time taken for generating response: {query_time:.2f} seconds")

# Cleanup: Remove temporary directory and its contents
cleanup_directory("documents")
