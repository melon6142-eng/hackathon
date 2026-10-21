#!/bin/bash
set -e
cd ~/AI/daker/cnn/runs/robust_pairs
P=~/AI/daker/venv/bin/python
LOG=run.log
echo "=== START $(date) ===" | tee -a $LOG
# Force CPU / limit threads to leave MPS for dino12
export CUDA_VISIBLE_DEVICES=''
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Do NOT set MPS - scripts hardcode cpu
echo "=== measure_noise $(date) ===" | tee -a $LOG
caffeinate -i $P -u measure_noise.py >>$LOG 2>&1
echo "=== pair_design $(date) ===" | tee -a $LOG
caffeinate -i $P -u pair_design.py >>$LOG 2>&1
echo "=== proxy_bl1_dino CPU $(date) ===" | tee -a $LOG
caffeinate -i $P -u proxy_bl1_dino.py >>$LOG 2>&1
echo "=== DONE compute $(date) ===" | tee -a $LOG
