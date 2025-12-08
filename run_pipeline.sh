#!/bin/bash
# Simple wrapper to run Prefect pipeline in local mode

echo " Starting IMDB Sentiment Analysis Pipeline with Prefect..."
echo ""

# Set Prefect to local mode (no server)
export PREFECT_API_URL=""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
 source venv/bin/activate
fi

# Run the pipeline
python pipeline/prefect_flow.py

echo ""
echo " Pipeline execution finished!"
echo ""
echo " View results:"
echo " - Metrics: cat metrics.json"
echo " - MLflow UI: mlflow ui --port 5000"
