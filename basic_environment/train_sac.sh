#!/bin/bash
rm -r logs
rm -r models 
python3 train_sac.py > output.txt 2>&1 &

