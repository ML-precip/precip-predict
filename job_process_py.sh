#!/bin/bash

#SBATCH --mem=312G
#SBATCH --nodes=1    
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090:4

module load Workspace
module load Anaconda3
module load Python
module load CUDA
module load cuDNN


#srun python DNN_models_comparison.py 
srun python Random_forest.py
