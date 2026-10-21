#!/bin/zsh
# uni_none_final chain: (A) 6 missing validation runs, (B) fs42 retrain with ckpt + sha256 + test-from-ckpt, (C) eval + decision. Resumable (skips finished runs; vit_arm skips finished folds).
set -u; cd ~/AI/daker/uni_none_final; export HF_HUB_OFFLINE=1 PYTHONPATH=~/AI/daker/final_clean/pylib:$PWD/src; PY=~/AI/daker/venv/bin/python
P=progress.txt; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $P; }
log "CHAIN start"
for fs in 7 123 2026; do for s in 1 2; do t=val/uni_none_s${s}_fs$fs; [[ -f arms/$t/oof.npy ]] && continue
 log "RUN $t start"; caffeinate -i $PY src/vit_arm.py --chan orig --seed $s --fs $fs --samp uni --aug none --tag $t --nockpt > logs/${t//\//_}.log 2>&1; log "RUN $t rc=$?"; done; done
log "STEP A done"
for s in 42 1 2; do t=ckpt/uni_none_s$s; if [[ ! -f arms/$t/oof.npy ]]; then
 log "RUN $t (ckpt) start"; caffeinate -i $PY src/vit_arm.py --chan orig --seed $s --fs 42 --samp uni --aug none --tag $t > logs/${t//\//_}.log 2>&1; log "RUN $t rc=$?"; fi
 [[ -f arms/$t/ckpt_check.json ]] || { caffeinate -i $PY src/infer_ckpt.py arms/$t ~/AI/daker/cnn/runs/bal_d4/arms/uni_none_s$s > logs/infer_s$s.log 2>&1; log "INFER $t rc=$?"; }; done
log "STEP B done"
$PY src/build_csv.py ckpt > logs/build_csv_ckpt.log 2>&1; log "CSV ckpt rc=$?"
$PY src/eval.py > logs/eval.log 2>&1; log "EVAL rc=$? -> eval.md, decision.md"
log "ALL DONE"
