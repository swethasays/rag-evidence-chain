#!/bin/bash

# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready before starting the UI
echo "Waiting for API to be ready..."
API_READY=0
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready."
        API_READY=1
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "API process exited unexpectedly. Aborting."
        exit 1
    fi
    sleep 1
done

if [ "$API_READY" -eq 0 ]; then
    echo "API did not become ready within 60 seconds. Aborting."
    kill "$API_PID" 2>/dev/null
    exit 1
fi

# Start Streamlit in foreground
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0
