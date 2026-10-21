#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
for RS in 7 123 2026; do
 until [ -f pre_rs${RS}_f0.pt ] && [ -f pre_rs${RS}_f1.pt ] && [ -f pre_rs${RS}_f2.pt ] && [ -f pre_rs${RS}_f3.pt ] && [ -f pre_rs${RS}_f4.pt ]; do sleep 60; done
 [ -f oof_frozen_pre_rs$RS.npy ] || caffeinate -i $P frozen_eval.py $RS > frozen_$RS.log 2>&1; echo FZDONE $RS $(date) >> fz_wait.log
done
