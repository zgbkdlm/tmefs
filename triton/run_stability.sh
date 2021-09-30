#!/bin/bash

#SBATCH -c 12
#SBATCH --time=4:00:00
#SBATCH --mem=8G

# Add --constraint=csl in order to compare running time

filter=$1
smoother=$2
sigma=$3

cd $WRKDIR/tmefs

module load anaconda
source ./venv/bin/activate

cd triton

if [ ! -d "./results" ]
then
    echo "Folder results does not exists. Trying to mkdir"
    mkdir ./results
fi

python stability.py -filter $filter -smoother $smoother -sigma $sigma
