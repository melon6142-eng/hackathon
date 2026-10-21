#!/bin/bash
cd ~/AI/daker/cnn;P=../venv/bin/python;Q=runs/indepfold/queue.log
n=0;while true; do c=$(grep -c '^RSDONE ' $Q 2>/dev/null || echo 0); if [ "$c" != "$n" ]; then n=$c; $P runs/indepfold/indep_eval.py > runs/indepfold/eval_auto.log 2>&1; cp runs/indepfold/eval_auto.log runs/indepfold/eval_auto.md; echo "EVAL after RSDONE=$n $(date)" >> $Q; fi; grep -q INDEPDONE $Q && [ "$n" = "3" ] && break; sleep 120; done
echo EVALDONE >> $Q
