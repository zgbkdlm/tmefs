#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

cd $WRKDIR/tmefs

module load anaconda

# Create venv
mkdir venv
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip

#
pip install -r requirements.txt

python setup.py install

