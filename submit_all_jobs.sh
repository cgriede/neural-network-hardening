#!/bin/bash
#SBATCH --job-name=BatchJobSubmitter
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=job_submit_log.txt
#SBATCH --error=job_submit_error.txt

# Base directory for job directories is the directory where this script is located
JOB_DIR_BASE=$(pwd)

# Iterate over each subdirectory in the job directory base
for dir in "$JOB_DIR_BASE"/*/
do
    # Check if the submission_file.sh file exists in the subdirectory
    if [ -f "${dir}submission_file.sh" ]; then
        # Convert to UNIX line endings
        dos2unix "${dir}submission_file.sh"
        
        # Submit the job from the correct directory
        (cd "${dir}" && sbatch submission_file.sh)
        echo "Submitted job in directory: $dir"
    else
        echo "No submission_file.sh found in directory: $dir"
    fi
done
