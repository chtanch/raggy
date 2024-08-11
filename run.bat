@echo off
if not exist venv (
    echo "venv does not exist. Exiting.."
    exit /b 1
)
call venv\Scripts\activate
streamlit run src\app.py