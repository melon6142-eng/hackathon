#!/bin/zsh
# usage: chain.sh chan  -> runs seeds 42(orig tag),1,2 sequentially
cd ~/AI/daker/cnn/runs/repro_vit; export HF_HUB_OFFLINE=1
c=$1
for s in 42 1 2; do
  t=dinoB_${c}; [ $s != 42 ] && t=dinoB_${c}_s$s
  echo "$(date) START $t" >> progress.txt
  caffeinate -i ~/AI/daker/venv/bin/python dino_ft_repro.py --arch vit_base_patch14_dinov2.lvd142m --size 252 --chan $c --unfreeze 4 --epochs 30 --bs 16 --lr 5e-05 --weight 0.0 --tag $t --seed $s > logs_$t.out 2>&1; r=$?
  echo "$(date) TRAINED $t rc=$r" >> progress.txt
  ~/AI/daker/venv/bin/python infer_from_ckpt.py $t $c >> logs_$t.out 2>&1
  echo "$(date) INFER $t rc=$?" >> progress.txt
done
echo "$(date) CHAIN $c DONE" >> progress.txt
