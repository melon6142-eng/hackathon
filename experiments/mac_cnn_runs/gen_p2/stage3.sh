#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
pre(){ RS=$1;f=$2; [ -f pool_rs${RS}_f$f.npz ] || $P make_pool.py rs${RS}_f$f $RS $f >> stage3.log 2>&1
 [ -f pre_rs${RS}_f$f.pt ] || caffeinate -i $P pretrain.py --pool pool_rs${RS}_f$f.npz --out pre_rs${RS}_f$f.pt >> stage3.log 2>&1; }
for RS in 7 123 2026; do
 pre $RS 0 & pre $RS 1 & wait; pre $RS 2 & pre $RS 3 & wait; pre $RS 4
 caffeinate -i $P frozen_eval.py $RS > frozen_$RS.log 2>&1; echo RSDONE $RS $(date) >> stage3.log
done
