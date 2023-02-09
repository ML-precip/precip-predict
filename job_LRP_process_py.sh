#!/bin/bash

#SBATCH --mem=312G
#SBATCH --nodes=1    
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx2080ti:4

module load Workspace
module load Anaconda3
#module load Python
module load CUDA
module load cuDNN

#need to load my py36
source activate 
conda activate /storage/homefs/no21h426/.conda/envs/py36

srun python LRP_DNN.py
