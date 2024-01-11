#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME=M4-R

# Create a new Conda environment for R
conda create -n $CONDA_ENV_NAME r-base=3.6.*

# Activate the Conda environment
conda activate $CONDA_ENV_NAME

conda install -c conda-forge r-akima r-forecast r-qrnn r-e1071 r-randomForest r-genalg r-doParallel r-foreach r-quantreg r-quantregGrowth r-matrixStats r-nloptr

Rscript -e 'install.packages(c("monmlp", "stringr"), dependencies=TRUE, repos="https://ftp.osuosl.org/pub/cran/")'

# Deactivate the Conda environment
conda deactivate

echo "R environment $CONDA_ENV_NAME is ready with the required packages installed."
