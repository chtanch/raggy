@echo off
if not exist venv (
    echo "venv does not exist. Exiting.."
    exit /b 1
)
call venv\Scripts\activate
python experiments\qdrant_basic.py
python experiments\qdrant_multitenancy.py
python experiments\query_csv.py