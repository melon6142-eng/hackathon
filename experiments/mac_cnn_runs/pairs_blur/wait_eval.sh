#!/bin/bash
cd ~/AI/daker/cnn/runs/pairs_blur
while ! grep -q BLURDONE ../../run_blur.log; do sleep 60; done
caffeinate -i ~/AI/daker/venv/bin/python -u eval_blur.py > eval_blur.log 2>&1; echo EVALDONE>>eval_blur.log
