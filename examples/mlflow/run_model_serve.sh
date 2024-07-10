#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:8080

# Serve the production model from the model registry
mlflow models serve -m "models:/rf_apples_01@1"