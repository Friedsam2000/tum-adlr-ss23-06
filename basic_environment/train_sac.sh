#!/bin/bash
rm -r logs
rm -r models 
nohup python3 train_sac.py > output.txt 2>&1 &

