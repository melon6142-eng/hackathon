#!/bin/bash
set -e
cd ./runs
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=./venv/bin/python
echo "=== CAL1 $(date) ===" | tee cal/run.log
$PY cal/cal1.py 2>&1 | tee -a cal/cal1.log
echo "=== CAL2 $(date) ===" | tee -a cal/run.log
$PY cal/cal2.py 2>&1 | tee -a cal/cal2.log
echo "=== CAL3 $(date) ===" | tee -a cal/run.log
$PY cal/cal3.py 2>&1 | tee -a cal/cal3.log
$PY cal/make_report.py 2>&1 | tee -a cal/run.log
echo "=== ALL DONE $(date) ===" | tee -a cal/run.log
