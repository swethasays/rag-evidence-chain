#!/bin/bash

# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0