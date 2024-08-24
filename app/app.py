import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

st.set_page_config(
    page_title="RAGgy",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("RAGgy, powered by LlamaIndex")
input_dir = "./data/input/whitepapers-ai"

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    intro = f"Ask me a question about the docs in **{input_dir}**"
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": intro,
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    docs = reader.load_data()
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(
        model="phi3",
        temperature=0.2,
        system_prompt="""You are an expert in extracting information from documents. 
        Keep your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

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
