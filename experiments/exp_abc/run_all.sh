#!/bin/bash
set -e
cd ~/AI/daker/cnn/runs/exp_abc
P=~/AI/daker/venv/bin/python
LOG=~/AI/daker/cnn/runs/exp_abc/run.log
export DEV=mps
export BS=8
export NT=4
echo "=== EXP_ABC START $(date) ===" | tee -a "$LOG"
echo "=== A+B extract $(date) ===" | tee -a "$LOG"
caffeinate -i $P -u extract_ab.py >>"$LOG" 2>&1
echo "=== C diag $(date) ===" | tee -a "$LOG"
caffeinate -i $P -u diag_c.py >>"$LOG" 2>&1
echo "=== eval $(date) ===" | tee -a "$LOG"
caffeinate -i $P -u eval_abc.py >>"$LOG" 2>&1
echo "=== EXP_ABC DONE $(date) ===" | tee -a "$LOG"
