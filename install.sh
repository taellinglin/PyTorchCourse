#!/bin/bash
# Create a virtual environment
python -m venv .pytorchenv
# Activate the virtual environment
source .pytorchenv/bin/activate
# Install required Python packages
pip install -r requirements.txt
