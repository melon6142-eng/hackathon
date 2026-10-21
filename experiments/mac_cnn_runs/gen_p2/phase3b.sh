#!/bin/bash
# Phase3 확장: fold rs 123, 2026 (Phase4 4-seed 검증용). phase3.sh(42,7)와 병렬.
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
for RS in 123 2026; do for f in 0 1 2 3 4; do
 [ -f pool_pair_rs${RS}_f$f.npz ] || $P make_pairpool.py $RS $f >> phase3b.log 2>&1
 [ -f pp_rs${RS}_f$f.pt ] || caffeinate -i $P pretrain_pair.py --pool pool_pair_rs${RS}_f$f.npz --out pp_rs${RS}_f$f.pt >> phase3b.log 2>&1
done; done; echo P3BPREDONE $(date) >> phase3b.log
