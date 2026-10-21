#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
for RS in 42 7; do for f in 0 1 2 3 4; do
 [ -f pool_pair_rs${RS}_f$f.npz ] || $P make_pairpool.py $RS $f >> phase3.log 2>&1
 [ -f pp_rs${RS}_f$f.pt ] || caffeinate -i $P pretrain_pair.py --pool pool_pair_rs${RS}_f$f.npz --out pp_rs${RS}_f$f.pt >> phase3.log 2>&1
done; done; echo P3PREDONE >> phase3.log
