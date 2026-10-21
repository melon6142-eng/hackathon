#!/bin/zsh
set -u; cd ~/AI/daker/cnn/runs/bal_d4; export HF_HUB_OFFLINE=1 PYTHONPATH=~/AI/daker/final_clean/pylib:$PWD/src; PY=~/AI/daker/venv/bin/python
P=progress.txt; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $P; }
log "FINALIZE waiting for chain end"; until grep -qE "[0-9] CHAIN DONE$" $P; do sleep 60; done
$PY src/analyze.py > logs/analyze_final.log 2>&1; log "ANALYZE final rc=$?"
W=$($PY src/expand.py); log "EXPAND arms: [${W}]"
for a in ${=W}; do sp=${a%%_*}; au=${a#*_}; for fs in 7 123 2026; do t=${a}_s42_fs$fs; [[ -f arms_fs/$t/oof.npy ]] && continue
 mkdir -p arms_fs; log "RUN $t"; caffeinate -i $PY src/vit_arm.py --chan orig --seed 42 --fs $fs --samp $sp --aug $au --tag ../arms_fs/$t --nockpt > logs/$t.log 2>&1; log "RUN $t rc=$?"; done; done
$PY src/report.py > logs/report.log 2>&1; log "REPORT rc=$? -> report.md (box 복사는 에이전트가 CopyToBox로)"; log "ALL DONE"
