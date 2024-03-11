# ASO SWE & Environmental Data Integration for Streamflow Prediction

## Overview
This repository offers a comprehensive framework for processing and analyzing snow water equivalent (SWE) data from Airborne Snow Observatories (ASO). It integrates this data with a diverse array of environmental and control datasets, including PRISM climate data, the National Land Cover Database (NLCD), elevation, aspect, grade, and geographic coordinates (latitude and longitude). The primary goal is to leverage these integrated datasets to forecast ASO SWE for a 20-year period. The developed predictive models are further used to estimate expected streamflow from snowpack using the M4 model, offering valuable insights for water resource management and ecological research.

## Key Features

- **Data Integration**: Includes scripts and notebooks for merging ASO SWE data with environmental variables like PRISM climate records, NLCD land cover, and topographic factors (elevation, aspect, grade), along with geographical positioning data.

- **Historical Analysis**: Provides tools for analyzing historical trends in snow water equivalent data and related environmental factors over a specified period.

- **SWE Prediction**: Implements machine learning algorithms to forecast future SWE values over a 20-year horizon, based on historical data and environmental controls.

- **Streamflow Estimation**: Utilizes predicted SWE data to generate expected streamflow metrics, employing hydrological models that consider varying snowpack contributions to river and stream flow.

- **Visualization Tools**: Offers a suite of visualization utilities for mapping SWE distributions, environmental factors, and streamflow predictions, simplifying data and predictions interpretation.

## Getting Started

### 1. Download Necessary Files

#### Download ASO Data
```
mkdir -p data/ASO/SWE
wget -i ASO_SWE_download_list.txt -P data/ASO/SWE
wget -i ASO_SWE_download_list_2020-2023.txt -P data/ASO/SWE
```

#### Download PRISM
```
./download_prism.sh
```

#### Clone Repositories
```
git clone git@github.com:nrcs-nwcc/M4.git
git clone git@github.com:johnwoodill/ASO-M4.git
```

#### Setup Python Environment
```
pyenv install 3.10
pyenv local 3.10
python -m venv ASO_env
source ASO_env/bin/activate
pip install -r requirements.txt
```

#### Setup R Environment
```
./setup-R-environment.sh
```

#### Run Code Workflow
```
python 1-Data-Step.py
python 2-NN-ASO-SWE-Model.py
python 3-Predictions.py
python 4-M4-Data-Step.py
```

#### Generate Figures
```
Rscript 5-Figures.R
Rscript 6-M4-Figures.R
```
