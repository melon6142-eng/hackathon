#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
one(){ RS=$1;f=$2; until [ -f pool_rs${RS}_f$f.npz ]; do sleep 30; done; sleep 5
 [ -f pre_rs${RS}_f$f.pt ] && return; touch pre_rs${RS}_f$f.lock; caffeinate -i $P pretrain.py --pool pool_rs${RS}_f$f.npz --out pre_rs${RS}_f$f.pt >> pre_runner.log 2>&1; rm -f pre_rs${RS}_f$f.lock; }
for RS in 7 123 2026; do one $RS 0 & one $RS 1 & wait; one $RS 2 & one $RS 3 & wait; one $RS 4; done; echo PRERUNDONE >> pre_runner.log
