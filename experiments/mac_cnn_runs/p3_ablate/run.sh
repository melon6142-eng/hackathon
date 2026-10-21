#!/bin/bash
cd ~/AI/daker/cnn/runs/p3_ablate;P=../../../venv/bin/python
RS=${RS:-42}
arg(){ case $1 in f10) echo --frac 0.1;; f30) echo --frac 0.3;; reg) echo --lam 0;; inv) echo --wreg 0;; scr) echo --scramble 1;; esac; }
for V in ${VARS:-reg inv f30 f10 scr}; do
 for f in 0 1 2 3 4; do
  [ -f ${V}_rs${RS}_f$f.pt ] || { echo "$(date +%T) start $V rs$RS f$f" >> progress.txt; caffeinate -i $P pt_abl.py --pool pool_pair_rs${RS}_f$f.npz --out ${V}_rs${RS}_f$f.pt $(arg $V) >> run.log 2>&1; }
 done
 [ -f oof_${V}_rs${RS}.npy ] || $P fz_eval.py $RS $V >> eval_rs$RS.log 2>&1; echo "$(date +%T) DONE $V rs$RS: $(tail -1 eval_rs$RS.log)" >> progress.txt
done
echo "$(date +%T) ALLDONE rs$RS" >> progress.txt
$P report.py > report.log 2>&1
