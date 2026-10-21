#!/bin/bash
cd ~/AI/daker/cnn; P=../venv/bin/python
while pgrep -f "train.py --tag" >/dev/null; do sleep 60; done
$P dino_ft.py --tag dino_orig --chan orig
$P dino_ft.py --tag dino_trans --chan trans
$P dino_ft.py --tag dino_trans_w --chan trans --weight 1.0
echo DINODONE
