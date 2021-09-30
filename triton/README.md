# Readme

This folder contains the scripts that output the experiment results in the manuscript. 

Although the bash scripts in this folder are to be run in Triton (or any similar slurm-based computational clusters), Aalto University, 
you may still run the `.py` files by yourself.

# Files

1. `run.py`: Run filtering and smoothing and output the mean and std. of the RMSE from 1,000 Monte Carlo runs. 
2. `run.sh`: Submission file in Triton to get the Table 1 in the manuscript.
3. `run_stability.sh`: Submission file in Triton to get the Figure 1 in the manuscript.
4. `stability.py`: Run filtering and smoothing and output the estimated $E[||X_k - m^s_k||_2^2]$ from 10,000 Monte Carlo runs.
5. `submit.sh`: Lazy man's submission file.
6. `triton_setup_venv.sh`: Setup an venv in Triton.
7. `./results`: A folder containing the results.

# How to reproduce the results

Say, for example, that you want to get the numbers of GHF EM GHS TME-2 in Table 1. 
Then you need to run `python run.py -filter EM -smoother TME-2`. The results will be outputed in folder `./results`.

