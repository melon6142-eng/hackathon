#!/bin/bash
cd ~/AI/daker/cnn;P=../venv/bin/python;LOG=run_dino4.log
run(){ tag=$1;shift
  [ -f runs/$tag/oof.npy ] && { echo "SKIP $tag" >>$LOG;return; }
  pgrep -f "dino_ft[0-9]*\.py" >/dev/null && { echo "ABORT other dino running" >>$LOG;exit 1; }
  echo "=== START $tag $(date)" >>$LOG; $P dino_ft4.py --tag $tag "$@" >>$LOG 2>&1; echo "=== DONE $tag $(date)" >>$LOG; }
run d4_lds --loss lds
run d4_pad266 --res pad
run d4_binreg --loss binreg
run d4_rank --loss rank
run d4_crop140 --res crop --crop 140 --epochs 20 --bs 32
run d4_s448 --size 448 --bs 8 --epochs 20
echo DINO4DONE >>$LOG
