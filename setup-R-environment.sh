#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME=ASO-R

# Create a new Conda environment for R
conda create -n $CONDA_ENV_NAME r-base

# Activate the Conda environment
conda activate $CONDA_ENV_NAME

# Install required R packages using Rscript
Rscript -e 'install.packages(c("akima", "forecast", "qrnn", "e1071", "randomForest", "monmlp", "genalg", "stringr", "doParallel", "foreach", "quantreg", "quantregGrowth", "matrixStats"), dependencies=TRUE)'

# Deactivate the Conda environment
conda deactivate

echo "R environment $CONDA_ENV_NAME is ready with the required packages installed."
