#!/bin/bash
cd ~/AI/daker/cnn/runs/struct_aux
while [ $(grep -c QDONE progress.log) -lt 2 ]; do [ -f aux0_s42/oof.npy -a -f aux0.3_s42/oof.npy -a ! -f ev_s42.txt ] && ../../../venv/bin/python ev.py > ev_s42.txt 2>&1; sleep 60; done
{ echo "# struct_aux 결과 (fold rs42, ViT-B orig 설정, aux 0.3 vs 대조 aux 0)"; echo; echo '```'; ../../../venv/bin/python ev.py; echo '```'; echo; echo "aux 타깃: $(tr '\n' ' ' < aux_names.txt)"; echo "표준화: fold별 train idx 통계만. test·합성·Phase2/3 가중치 미사용."; } > report.md 2>&1
echo FINDONE >> progress.log
