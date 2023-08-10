#!/bin/bash

# Install system dependencies
sudo apt update
sudo apt-get install -y python3-pip  swig python3-opencv python3-venv

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# Ensure that pip, setuptools, and wheel are up-to-date
pip install --upgrade pip setuptools wheel

# Install specific versions of PyTorch, torchvision, and torchaudio compatible with CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install Python dependencies
pip install -r requirements.txt
