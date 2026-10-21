#!/bin/zsh
# DAKER final compliant pipeline (end-to-end). usage: ./run_all.sh [main|extra|all]   (default all)
set -u; cd ~/AI/daker/final_clean; export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/pylib:$PWD/src; PY=~/AI/daker/venv/bin/python
P=progress.txt; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $P; }
mode=${1:-all}; mkdir -p feat bl1 vit logs
if [[ $mode != extra ]]; then
 log "STEP0 env/weights sha256"; $PY - > logs/env.txt <<'PYE'
import torch,timm,sklearn,numpy,cv2,skimage,platform,glob,hashlib
print('torch',torch.__version__,'timm',timm.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'cv2',cv2.__version__,'skimage',skimage.__version__,platform.platform())
for p in glob.glob('~/.cache/huggingface/hub/models--timm--vit_base_patch14_dinov2.lvd142m/snapshots/*/*'):
    import os; rp=os.path.realpath(p); print('timm',p,hashlib.sha256(open(rp,'rb').read()).hexdigest())
PYE
 [[ -f feat/dino_test.npy ]] || { log "STEP1 DINO feats"; caffeinate -i $PY src/feat_dino.py > logs/feat_dino.log 2>&1; log "STEP1 rc=$?"; }
 [[ -f feat/hand_test.csv ]] || { log "STEP2 hand feats"; $PY src/feat_hand.py > logs/feat_hand.log 2>&1; log "STEP2 rc=$?"; }
 log "STEP3 BL1 fs42 + test"; $PY src/bl1.py 42 > logs/bl1_42.log 2>&1; log "STEP3 rc=$?"
 log "STEP4 ViT x6 fs42 (2 parallel chains)"
 for c in orig trans; do ( for s in 42 1 2; do caffeinate -i $PY src/vit.py --chan $c --seed $s --fs 42 --tag ${c}_s$s > logs/vit_${c}_s$s.log 2>&1; log "VIT ${c}_s$s rc=$?"; done ) & done; wait
 log "STEP5 blend"; $PY src/blend.py > logs/blend.log 2>&1; log "STEP5 rc=$? MAIN DONE"
fi
if [[ $mode != main ]]; then
 log "EXTRA BL1 fs 7/123/2026"; $PY src/bl1.py 7,123,2026 > logs/bl1_extra.log 2>&1 &
 for c in orig trans; do ( for fs in 7 123 2026; do for s in 42 1 2; do caffeinate -i $PY src/vit.py --chan $c --seed $s --fs $fs --tag ${c}_s${s}_fs$fs --nockpt > logs/vit_${c}_s${s}_fs$fs.log 2>&1; log "VIT ${c}_s${s}_fs$fs rc=$?"; done; done ) & done; wait
 $PY src/blend.py > logs/blend.log 2>&1; log "EXTRA DONE rc=$?"
fi
