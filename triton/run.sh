#!/bin/bash

#SBATCH -c 10
#SBATCH --time=0:20:00
#SBATCH --mem=8G

# Add --constraint=csl in order to compare running time

filter=$1
smoother=$2

cd $WRKDIR/tmefs

module load anaconda
source ./venv/bin/activate

cd triton

if [ ! -d "./results" ]
then
    echo "Folder results does not exists. Trying to mkdir"
    mkdir ./results
fi

python run.py -filter $filter -smoother $smoother
