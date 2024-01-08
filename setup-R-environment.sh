#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME=M4-R

# Create a new Conda environment for R
conda create -n $CONDA_ENV_NAME r-base

# Activate the Conda environment
conda activate $CONDA_ENV_NAME

conda install -c conda-forge r-akima r-forecast r-qrnn r-e1071 r-randomForest r-genalg r-stringr r-doParallel r-foreach r-quantreg r-quantregGrowth r-matrixStats

Rscript -e 'install.packages("monmlp", dependencies=TRUE, repos="https://ftp.osuosl.org/pub/cran/")'


# Deactivate the Conda environment
conda deactivate

echo "R environment $CONDA_ENV_NAME is ready with the required packages installed."
