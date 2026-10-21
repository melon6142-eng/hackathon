#!/bin/bash
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python
[ -f fitted_params_v5.json ] || $P fit_tex.py > fit_tex.log 2>&1
[ -f pool_all.npz ] || $P make_pool.py all >> stage1.log 2>&1
caffeinate -i $P realism.py > realism.log 2>&1
for f in 0 1 2 3 4; do [ -f pool_rs42_f$f.npz ] || $P make_pool.py rs42_f$f 42 $f >> stage1.log 2>&1; done
echo STAGE1DONE >> stage1.log
