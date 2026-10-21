#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
for RS in 42 7; do until ls pp_rs${RS}_f{0,1,2,3,4}.pt >/dev/null 2>&1; do sleep 60; done
 [ -f oof_frozen_pp_rs$RS.npy ] || caffeinate -i $P frozen_eval_pp.py $RS > frozen_pp_$RS.log 2>&1; echo P3FZ $RS $(date) >> p3_wait.log; done
