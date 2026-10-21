#!/bin/bash
# usage: run.sh TAG "RS..." [extra args]
cd ~/AI/daker/cnn/runs/ssl_train;P=../../../venv/bin/python;TAG=$1;shift;RSS=$1;shift
for RS in $RSS; do for f in 0 1 2 3 4; do
 [ -f $TAG/rs${RS}_f$f/DONE ] || caffeinate -i $P vicreg_pre.py --rs $RS --fold $f --tag $TAG "$@" >> run_$TAG.log 2>&1
 echo "rs$RS f$f done $(date)" >> progress_$TAG.txt
done; $P ev.py $TAG $RS >> ev_$TAG.log 2>&1; done; echo ALLDONE >> progress_$TAG.txt
