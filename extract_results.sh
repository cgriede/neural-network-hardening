#!/bin/bash

# Create a 'results' directory in the parent folder
mkdir -p results

# Loop over each subdirectory in the current directory
for dir in */; do
    # Check if the 'archive_dev' folder exists in the current subdirectory
    if [ -d "$dir/archive_dev" ]; then
        # Create a corresponding directory in the 'results' folder
        mkdir -p "results/${dir%/}"
        
        # Copy the contents of 'archive_dev' to the 'results' subdirectory
        cp -r "$dir/archive_dev/"* "results/${dir%/}/"
    fi
done

echo "Extraction complete. Check the 'results' folder."
