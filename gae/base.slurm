#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Sep 14 2020 16:39:51 GMT+1000 (Australian Eastern Standard Time)

## Use GPU
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:1

# CPU-only (note: constrain to faster Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz)
#SBATCH --qos=normal
#SBATCH --partition=physical
#SBATCH --constraint=physg5

#SBATCH --mem=110G

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="Autoencoder"

# The project ID which this job should run under:
#SBATCH --account="punim1343"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-0:00:00

# Custom output name
#SBATCH --output=./output/%j-slurm.out

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
 echo "You need to submit your job to the queuing system with sbatch"
 exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
source /usr/local/module/spartan_old.sh
module load scikit-learn/0.18-intel-2016.u3-Python-2.7.9
module load Tensorflow/0.12.1-intel-2017.u2-Python-2.7.13-GPU

# The job command(s):
python2 run.py
