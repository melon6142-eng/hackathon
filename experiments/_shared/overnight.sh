#!/bin/bash
# 밤샘 드라이버: 07:30(박스 시계)에 정지. 스레드 3 제한.
cd ./runs; PY=./venv/bin/python; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
END=$(date -d "07:30" +%s); [ $END -lt $(date +%s) ] && END=$(date -d "tomorrow 07:30" +%s)
left(){ echo $(( END - $(date +%s) )); }
J=./out/journal.md
log(){ echo "$(date +%H:%M) $*" | tee -a overnight.log; }
run(){ local L=$(left); [ $L -le 120 ] && { log "deadline, skip $1"; exit 0; }; log "START $1"; timeout $L $PY stage.py "$@" >> overnight.log 2>&1; log "DONE $1 rc=$?"; }
extract(){ local f=dino/$1_$2_L12ms_test.npy; [ -f $f ] && return; local L=$(left); [ $L -le 600 ] && { log "deadline, skip extract $1 $2"; exit 0; }
  pgrep -f "feat_dino_ml.py $1 $2" >/dev/null && { log "wait existing $1 $2"; while pgrep -f "feat_dino_ml.py $1 $2" >/dev/null; do sleep 30; done; [ -f $f ] && return; }
  log "EXTRACT $1 $2"; timeout $(left) $PY dino/feat_dino_ml.py $1 $2 12 > dino/log_$1_$2_L12.txt 2>&1; log "EXTRACT done $1 $2"; }
echo -e "\n### 자동 드라이버 단계 (overnight.sh, 시작 $(date +%H:%M))\n| 단계 | 구성 | CV RMSE | hi | lo | 비고 | 시각 |\n|---|---|---|---|---|---|---|" >> $J
# 기존 큐(b252 L12 → s252 L12)는 그대로 두고 결과 대기
while pgrep -f eval_dino2.py >/dev/null; do sleep 30; done
extract vitb14 252
B=vitb14_252_L12ms
run S1_b_L9-12_hx "$B:9-12:ms" hx            # BL1 재현, HX 선택 fold 내부(낙관 제거)
run S2_b_L5-8 "$B:5-8:ms"
run S3_b_L5-12_hx "$B:5-12:ms" hx
run S4_b_L1-12_hx "$B:1-12:ms" hx
run S5_b_L5-8+L9-12_hx "$B:5-8:ms;$B:9-12:ms" hx
extract vits14 252
S=vits14_252_L12ms
run S6_s_L5-12_hx "$S:5-12:ms" hx
run S7_b+s_hx "$B:5-12:ms;$S:5-12:ms" hx
extract vits14 518
S5=vits14_518_L12ms
run S8_s518_L5-12_hx "$S5:5-12:ms" hx
run S9_b252+s518_hx "$B:5-12:ms;$S5:5-12:ms" hx
run S10_b252+s252+s518_hx "$B:5-12:ms;$S:5-12:ms;$S5:5-12:ms" hx
extract vitb14 224
run S11_b224+b252+s518_hx "vitb14_224_L12ms:5-12:ms;$B:5-12:ms;$S5:5-12:ms" hx
log "ALL DONE"; echo "| 종료 | 드라이버 전 단계 완료 $(date +%H:%M) | | | | | |" >> $J
