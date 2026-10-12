#!/bin/bash
cd ~/AI/daker/cnn; P=../venv/bin/python
$P train.py --tag cnx_full --mode full --epochs 50
$P train.py --tag cnx_patch --mode patch --crop 128 --epochs 40
$P train.py --tag cnx_full_w --mode full --epochs 50 --weight 1.0
$P train.py --tag effb0_full --arch efficientnet_b0.ra_in1k --mode full --epochs 50 --lr 5e-4
echo ALLDONE
