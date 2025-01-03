#!/bin/bash

if [ -z "$SERVICE_NAME" ]; then
    echo "SERVICE_NAME environment variable is not set"
    exit 1
fi

echo "Starting service: $SERVICE_NAME"
python -m "microservices.$SERVICE_NAME"