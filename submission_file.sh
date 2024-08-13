#!/bin/bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=23:59:00
#SBATCH --partition=batch

module purge
module load stack/2024-06
module load gcc/12.2.0
module load libjpeg-turbo/3.0.0
module load abaqus/2023


#activate environment
source /cluster/home/cgriede/nnet/bin/activate

dos2unix model_dev.py
dos2unix utils.py

# Run your script
python model_dev.py --cpus $SLURM_CPUS_PER_TASK