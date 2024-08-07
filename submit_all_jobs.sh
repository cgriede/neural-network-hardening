#!/bin/bash

# SLURM settings for the script itself
#SBATCH --job-name=BatchJobSubmitter
#SBATCH --time=00:10:00

# Base directory for job directories is the directory where this script is located
JOB_DIR_BASE=$(pwd)

# Iterate over each subdirectory in the job directory base
for dir in "$JOB_DIR_BASE"/*/
do
    # Check if the submission.sh file exists in the subdirectory
    if [ -f "${dir}submission.sh" ]; then
        # Submit the job
        sbatch "${dir}submission.sh"
        echo "Submitted job in directory: $dir"
    else
        echo "No submission.sh found in directory: $dir"
    fi
done
