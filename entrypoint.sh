#!/bin/sh

# Install any new dependencies
pip install -r requirements.txt

# Run the application
exec uvicorn fastapi_app:app --host 0.0.0.0 --port 8080 --reload