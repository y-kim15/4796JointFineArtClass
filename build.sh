#!/bin/bash

# Uncomment to log all commands as they run
set -x

# Destroy the existing virtual environment
echo "Deleting existing virtual environment..."
rm -rf venv

# Create new virtualenv
echo "Create virtual environment..."
/usr/local/python/bin/python3 -m venv venv || { echo "error creating venv" ; exit 1; }

# Tell TensorFlow where to find the Cuda libraries (wheels are linked against Cuda 10.0)
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64' >> venv/bin/activate

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip || { echo "error upgrading pip"; exit 1; }

# Install required packages
echo "Installing modules from requirements.txt..."
pip3 install --upgrade --requirement requirements.txt || { echo "error installing requirements"; exit 1; }

# Fini
echo
echo "Type: 'source venv/bin/activate' (no quotes) in a terminal to activate the venv."
echo
