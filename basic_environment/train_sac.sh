#!/bin/bash
rm -r logs
rm -r models
rm output.txt
nohup bash python3 train_sac.py > output.txt 2>&1 &

