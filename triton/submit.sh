#!/bin/bash

for filter in "EKF" "EM" "TME-2" "TME-3"; do
    for smoother in "EKS" "EM" "TME-2" "TME-3"; do
        sbatch run.sh $filter $smoother
    done
done
