import streamlit as st
import rag
from llama_index.core import VectorStoreIndex
import os

st.set_page_config(
    page_title="RAGgy",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("RAGgy, powered by LlamaIndex")

# Hardcode the input file directory as streamlit does not provide full path.
directory = "C:\\Intel\\raggy\\data\\multi-types"

uploaded_files = st.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
)


if uploaded_files:
    if (
        "messages" not in st.session_state.keys()
    ):  # Initialize the chat messages history
        intro = f"Ask me a question about the uploaded docs"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": intro,
            }
        ]
    input_files = []
    for file in uploaded_files:
        input_files.append(os.path.join(directory, file.name))
    docs = rag.load_data(input_files)
    st.write(docs)
    index = rag.load_index(docs)

    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = rag.initialize_chat_engine(index)

    if prompt := st.chat_input(
        "Ask a question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Write message history to UI
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)
