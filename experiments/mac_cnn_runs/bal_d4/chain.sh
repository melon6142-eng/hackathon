#!/bin/zsh
set -u; cd ~/AI/daker/cnn/runs/bal_d4; export HF_HUB_OFFLINE=1 PYTHONPATH=~/AI/daker/final_clean/pylib:$PWD/src; PY=~/AI/daker/venv/bin/python
P=progress.txt; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $P; }
log "CHAIN start (single MPS chain alongside final_clean)"
[[ -f arms/emb_pre/fold4_emb.npz ]] || { caffeinate -i $PY src/emb_ckpt.py > logs/emb.log 2>&1; log "EMB base/pre rc=$?"; }
for s in 42 1 2; do for arm in uni_none:uni:none bal_d4b:bal:d4b bal_none:bal:none uni_d4:uni:d4; do
 t=${arm%%:*}; r=${arm#*:}; sp=${r%%:*}; au=${r#*:}; tag=${t}_s$s
 [[ -f arms/$tag/oof.npy ]] && continue
 log "RUN $tag start"; caffeinate -i $PY src/vit_arm.py --chan orig --seed $s --fs 42 --samp $sp --aug $au --tag $tag --nockpt > logs/$tag.log 2>&1; log "RUN $tag rc=$?"
done; [[ -f src/analyze.py ]] && { $PY src/analyze.py > logs/analyze_s$s.log 2>&1; log "ANALYZE after s$s rc=$?"; }; done
log "CHAIN DONE"
