#!/bin/bash
rm -r logs
rm -r models 
rm output.txt
nohup python3 train_pearl_gce.py > output.txt 2>&1 &

