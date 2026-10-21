#!/bin/bash
# usage: q.sh W ; runs seeds 42 1 2 at fold_rs 42
cd ~/AI/daker/cnn/runs/struct_aux;P=../../../venv/bin/python;W=$1
for S in 42 1 2; do t=aux${W}_s${S}; [ -f $t/oof.npy ] && continue
 echo "START $t $(date)">>progress.log; caffeinate -i $P sa_ft.py --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --chan orig --seed $S --fold_rs 42 --aux_w $W --tag $t >>$t.log 2>&1; echo "DONE $t $(date) $(tail -1 $t.log)">>progress.log; done
echo "QDONE $W">>progress.log
