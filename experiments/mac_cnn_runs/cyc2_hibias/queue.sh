#!/bin/bash
# wait for ssl_train(vicreg_pre) & p3_ablate(pt_abl/driver/run.sh) to finish, then run cyc2. Output only in this folder.
cd ~/AI/daker/cnn;P=../venv/bin/python;Q=runs/cyc2_hibias;LOG=$Q/queue.log
busy(){ pgrep -f 'vicreg_pre.py|pt_abl.py|p3_ablate|ssl_train|dino_ft' >/dev/null; }
echo "=== WAIT $(date)">>$LOG; n=0
while [ $n -lt 3 ]; do if busy; then n=0; else n=$((n+1)); fi; sleep 300; done
echo "=== GPU FREE, START $(date)">>$LOG
A="--arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --chan orig --seed 42"
run(){ t=$1; [ -f $Q/$t/oof_x.npy ] && { echo "=== SKIP $t">>$LOG; return; }
 echo "=== START $t $(date)">>$LOG; caffeinate -i $P $Q/ft.py $A ${@:2} --tag $t >>$LOG 2>&1; echo "=== DONE $t $(date)">>$LOG; $P $Q/ev.py > $Q/eval.log 2>&1; }
for V in base log hiw1 log_hiw1; do for RS in 7 123 2026; do
 case $V in base) X="";; log) X="--tgt log";; hiw1) X="--weight 1.0";; log_hiw1) X="--tgt log --weight 1.0";; esac
 run rs${RS}_${V} --fold_rs $RS $X; done; done
echo "CYC2DONE $(date)">>$LOG
