#!/bin/bash
cd ~/AI/daker/cnn/runs/ssl_train;P=../../../venv/bin/python
until grep -q ALLDONE progress_v3.txt 2>/dev/null; do sleep 30; done
$P report.py
BEST=$($P -c "
import json
b=min(['v1','v2','v3'],key=lambda t:json.load(open(f'ev_{t}_rs42.json'))['eq4(+SSL)']['rmse'])
r=json.load(open(f'ev_{b}_rs42.json'));d=r['eq4(+SSL)']['rmse']-r['eq3']['rmse']
print(b if (d<=-0.03 or r['SSL frozen']['rmse']<=13.0) else 'NONE')")
echo "best=$BEST $(date)" >> chain.log
if [ "$BEST" = NONE ]; then
 echo "seed 42에서 세 변형 모두 기준 미달(eq4 개선 <0.03, SSL 단독 >13.0). 4 seed 확장 없이 중단. 채택 안 함, 테스트 CSV 미생성." > verdict.txt
else
 A=$(grep -o -- "--ep.*" <<< "$(case $BEST in v1) echo '--ep 8 --lr 3e-5 --nblk 4';; v2) echo '--ep 30 --lr 5e-5 --nblk 6';; v3) echo '--ep 30 --lr 1e-4 --nblk 4';; esac)")
 ./run.sh $BEST "7 123 2026" $A; $P report.py
 echo "$BEST 4seed 완료 — 판정은 report.md 표 참고 (수동 확인 필요)" > verdict.txt
fi
$P report.py; echo CHAINDONE >> chain.log
