#!/bin/bash
# Sequential only. Skip if tag already has oof.npy. No parallel dino_ft.
set -e
cd ~/AI/daker/cnn
P=../venv/bin/python
LOG=run_dino3.log
run(){
  tag=$1; shift
  if [ -f "runs/$tag/oof.npy" ]; then echo "SKIP $tag already done" | tee -a $LOG; return; fi
  # refuse if another dino_ft is running
  if pgrep -f "dino_ft.py" >/dev/null; then echo "ABORT: another dino_ft running" | tee -a $LOG; exit 1; fi
  echo "=== START $tag $(date) ===" | tee -a $LOG
  $P dino_ft.py --tag "$tag" "$@" 2>&1 | tee -a $LOG
  echo "=== DONE $tag $(date) ===" | tee -a $LOG
}
# (1) extra seeds for ViT-B 252 orig & trans (fold split fixed; seed=train RNG)
run dinoB_orig_s1 --chan orig --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --seed 1
run dinoB_orig_s2 --chan orig --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --seed 2
run dinoB_trans_s1 --chan trans --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --seed 1
run dinoB_trans_s2 --chan trans --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 5e-5 --seed 2
# (2) unfreeze 6 / 8 on trans, lower lr
run dinoB_trans_u6 --chan trans --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 3e-5 --unfreeze 6 --seed 42
run dinoB_trans_u8 --chan trans --arch vit_base_patch14_dinov2.lvd142m --size 252 --bs 16 --lr 3e-5 --unfreeze 8 --seed 42
echo DINO3DONE | tee -a $LOG
