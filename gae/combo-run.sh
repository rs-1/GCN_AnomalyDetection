#!/bin/bash

DATASETS=(
    'node-capture20110815-sample_10-matlab'
)

for D in "${DATASETS[@]}"; do
    sbatch <(
        sed                 \
        -e s/_1DATASET_/$D/ \
        base.slurm
    )
done
