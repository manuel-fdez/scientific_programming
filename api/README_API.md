# Scientific Programming - Collaborator 6 API (FastAPI)

This folder contains the FastAPI deployment service for the trained model.

## Setup

From the repository root:

```bash
cd ~/scientific_programming
source .venv/bin/activate
pip install -r api/requirements.txt

uvicorn api.app.main:app --reload --host 127.0.0.1 --port 8000

