#!/bin/bash
cd ~/AI/daker/cnn;P=../venv/bin/python;LOG=run_dino6.log
echo "wait for dino4/dino5 $(date)" >>$LOG
while screen -ls | grep -qE '\.dino[45][[:space:]]' || pgrep -f "dino_ft[45]\.py" >/dev/null; do sleep 60; done
run(){ tag=$1;shift
  [ -f runs/$tag/oof.npy ] && { echo "SKIP $tag" >>$LOG;return; }
  pgrep -f "dino_ft[0-9]*\.py" >/dev/null && { echo "ABORT other dino running" >>$LOG;exit 1; }
  echo "=== START $tag $(date)" >>$LOG; caffeinate -i $P dino_ft6.py --tag $tag "$@" >>$LOG 2>&1; echo "=== DONE $tag $(date)" >>$LOG; }
# v1 (pairwise rank) skipped: identical to d4_rank (nCV OOF 13.03, hi -26.0)
run d6_maskcoh --chan maskcoh --seed 42
run d6_maskaux --chan mask --aux 0.3 --seed 42
echo DINO6DONE >>$LOG
