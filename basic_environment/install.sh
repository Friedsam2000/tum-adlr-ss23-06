#!/bin/bash

# Install system dependencies
sudo apt update
# sudo apt-get install -y python3-pip  swig python3-opencv python3-venv

# Create a virtual environment and activate it
python3 -m venv venv_t
source venv_t/bin/activate

# Ensure that pip, setuptools, and wheel are up-to-date
pip install --upgrade wheel
pip install setuptools==65.5.0 pip==21
pip install importlib-metadata==4.13.0

# Install Python dependencies
# pip install -r requirements.txt
pip install -e ~/stable-baselines3 [docs,tests,extra]
