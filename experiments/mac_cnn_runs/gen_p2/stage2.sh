#!/bin/bash
# rs=$1 : fold별 사전학습(2개씩 병렬) -> 경도 FT 3 seed 병렬
cd ~/AI/daker/cnn/runs/gen_p2;P=../../../venv/bin/python;RS=${1:-42}
until grep -q STAGE1DONE stage1.log 2>/dev/null || [ "$RS" != 42 ]; do sleep 30; done
pre(){ f=$1; while [ -f pre_rs${RS}_f$f.lock ]; do sleep 30; done; [ -f pool_rs${RS}_f$f.npz ] || $P make_pool.py rs${RS}_f$f $RS $f >> stage2_$RS.log 2>&1
 [ -f pre_rs${RS}_f$f.pt ] || caffeinate -i $P pretrain.py --pool pool_rs${RS}_f$f.npz --out pre_rs${RS}_f$f.pt >> stage2_$RS.log 2>&1; }
pre 0 & pre 1 & wait; pre 2 & pre 3 & wait; pre 4
echo PREDONE $(date) >> stage2_$RS.log
A="--arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --chan orig"
for S in 42 1 2; do caffeinate -i $P ft_hard.py $A --seed $S --fold_rs $RS --init 'pre_rs{rs}_f{f}.pt' --tag pre_rs${RS}_s$S >> stage2_$RS.log 2>&1 & done; wait
echo FTDONE $(date) >> stage2_$RS.log
