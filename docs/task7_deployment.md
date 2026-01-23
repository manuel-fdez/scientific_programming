# Task 7 — Docker Deployment (Collaborator 7)

This guide explains how to containerize and run the FastAPI API locally using Docker, then test the deployed endpoints.

## Requirements
- Docker Desktop installed and running
- Repository cloned locally

## Project details (confirmed)
- FastAPI app: `api/app/main.py`
- Uvicorn app path: `api.app.main:app`
- Dependencies: `api/requirements.txt`
- Required artifacts:
  - `api/artifacts/features.json`
  - `api/artifacts/best_model.pkl`


## Quick start

### 1) Build the Docker image
From the repository root:

```bash
docker build -t scientific-api:1.0 .


### 2) Deploy locally (laptop as server)
docker rm -f scientific-api 2>/dev/null
docker run -d --name scientific-api --restart unless-stopped -p 8000:8000 scientific-api:1.0
docker ps


### 3) Test deployed API

Health:

curl -i http://localhost:8000/health
curl -i http://localhost:8000/schema/features

Docs (browser):
http://localhost:8000/docs

### 4) Predict:

curl -i -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{...}'
Notes

GET / returns 404 because no root endpoint is defined (expected).
