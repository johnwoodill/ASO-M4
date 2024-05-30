#!/bin/bash

# Instructions

# Make Script Executable: Save the script as process_M4.sh and then make it 
# executable with the command chmod +x process_M4.sh.
# Run the Script: Execute the script by typing 

# ./process_M4.sh /path/to/your/base/directory WatershedName 
# ./process_M4.sh ~/Projects/M4/examples/ Tuolumne_Watershed
# ./process_M4.sh ~/Projects/M4/examples/ Blue_Dillon_Watershed
# ./process_M4.sh ~/Projects/M4/examples/ Dolores_Watershed
# ./process_M4.sh ~/Projects/M4/examples/ Conejos_Watershed

# in your terminal. Replace /path/to/your/base/directory with the actual 
# path where the directories are located and WatershedName with the actual 
# name of your watershed.

# -----------------------------------------------------------------------------

# conda activate M4-R

# Check if two arguments are given (base directory and watershed name)
if [ $# -ne 2 ]; then
    echo "Usage: $0 [base directory] [watershed name]"
    exit 1
fi

# Assign the first argument as the base directory
BASE_DIR=$1

# Assign the second argument as the watershed name
WATERSHED_NAME=$2

# Define the array of directories based on the watershed name
declare -a directories=(
"${WATERSHED_NAME}_aso_swe"
"${WATERSHED_NAME}_aso_swe_total"
"${WATERSHED_NAME}_baseline"
"${WATERSHED_NAME}_baseline_aso_swe"
"${WATERSHED_NAME}_baseline_aso_swe_temp_precip"
"${WATERSHED_NAME}_aso_temp_precip"
)

# Loop through each directory
for dir in "${directories[@]}"; do
    # Construct the full directory path
    DIR_PATH="${BASE_DIR}/${dir}"

    # Check if the directory exists
    if [ -d "$DIR_PATH" ]; then
        echo "Entering $DIR_PATH"
        cd "$DIR_PATH"

        # Remove files with the prefix "Rplot"
        rm Rplot*

        # Run the R script
        Rscript MMPE-Main_MkII.R

        # Return to the base directory (important if the directories are not nested)
        cd "$BASE_DIR"
    else
        echo "Directory $DIR_PATH does not exist."
    fi
done
