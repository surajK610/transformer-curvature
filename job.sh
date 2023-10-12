#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=24:00:00

# Ask for the GPU partition and 1 GPU
# skipping this for now.
#SBATCH -p 3090-gcondo --gres=gpu:1

# Use more memory (8GB) and correct partition.
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J norm_loss_comparison

# Specify an output file
#SBATCH -o ./out/%x-%a.out
#SBATCH -e ./err/%x-%a.out

#SBATCH -a 0-4%15


# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
module load python/3.9.0 cuda/11.1.1 gcc/10.2
source venv/bin/activate

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_resids_sd_1 --size-dict 1 --resid yes
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_outs_sd_1 --size-dict 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_resids_sd_2 --size-dict 2 --resid yes
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_outs_sd_2 --size-dict 2
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_resids_sd_2 --size-dict 3 --resid yes
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_outs_sd_2 --size-dict 3
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_resids_sd_2 --size-dict 4 --resid yes
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ];
then
python3 -m src.experiments.lookup --size 50 --name layer_wise_analysis_50_outs_sd_2 --size-dict 4
fi