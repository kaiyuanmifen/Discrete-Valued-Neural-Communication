#!/bin/bash 

echo "copying logs from all folders"

cd checkpoints/
for f in */log.txt; do
    cp -v "$f" /n/scratch3/users/d/dl249/GNN/c-swm_dianbo_h_15/Eval_results/TrainingLogs/"${f//\//_}"
done

echo "done"