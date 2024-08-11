@echo off
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull phi3
ollama pull nomic-embed-text