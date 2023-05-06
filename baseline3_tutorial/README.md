# Baseline 3 Tutorial

This project is set up to run in a Python virtual environment. Follow the instructions below to set up the environment, install dependencies, and run the code.

## Prerequisites

- Python 3.6 or higher

## Setup

1. Create a virtual environment and activate it.

For Linux and macOS:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
For windows:

   ```bash
   py -m venv venv
   venv\Scripts\activate.bat
   ```
2. Install the dependencies.

   ```bash
    pip install -r requirements.txt
    ```
   
3. Run the code.

   ```bash
   python main.py
   ```
   
4. Monitor the training progress in TensorBoard.

   ```bash
   tensorboard --logdir=runs
   ```
   
