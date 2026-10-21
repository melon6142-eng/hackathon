#!/bin/bash
# rs42 끝나면 rs7(신호 확인용: reg inv scr)
cd ~/AI/daker/cnn/runs/p3_ablate
while ! grep -q "ALLDONE rs42" progress.txt; do sleep 120; done
RS=7 VARS="reg inv scr f30" ./run.sh > nohup_rs7.out 2>&1
