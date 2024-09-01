# Conformal Time Series Decomposition

This repository contains the code and instructions for reproducing the experiments in the paper "Conformal time series decomposition with component-wise exchangeability".

[![arXiv](https://img.shields.io/badge/arXiv-2406.16766-b31b1b.svg)](https://arxiv.org/abs/2406.16766)


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Cite](#cite)

## Introduction

In this paper, we propose a novel method for decomposing time series data using conformal prediction and component-wise exchangeability. This README provides instructions on how to run the code and reproduce the experiments described in the paper.

## Installation

To run the code, you need to have the following version of Python installed:

- Python 3.10.13

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

The `runner.py` script can be executed on any dataset using the `--dataset` argument. The dataset is decomposed into trend, seasonal, and noise components. Predictions are made on each of these components, including the original time series, using a specified regressor (`--basemodel`). Prediction intervals are calculated for each component. You can specify the conformal method for each component using the following arguments:
- `--cp_method_original`
- `--cp_method_trend`
- `--cp_method_seasonal`
- `--cp_method_noise`

Results are automatically logged to Weights & Biases (wandb) for each component, as well as for the original and recomposed time series.

When specifying each basemodel or conformal method, you may need to set hyperparameters. See the `arg_parser` in `runner.py` for more details.

### Examples
Some examples with which some of our experimental results can be reproduced. 
#### Synthetic Dataset
Using the Linear model, running Enbpi on the trend component, BinaryPoint on the seasonal component, and CV+ on the noise component:

```sh
python src/runner.py --dataset synthetic --basemodel Linear --cp_method_trend enbpi --cp_method_seasonal local_cp --cp_method_noise cv_plus --local_cp_method periodic --use_region False --use_exponential False
```

#### Energy Dataset
Using the MLP model, running ACI on the trend component, BinaryLocal on the seasonal component, and CV+ on the noise component:

```sh
python src/runner.py --dataset energy-consumption --basemodel MLP --cp_method_trend aci --cp_method_seasonal local_cp --cp_method_noise cv_plus --local_cp_method periodic --use_region True --use_exponential False
```

#### Sales Dataset
Using the Linear model, running ACI on the trend component, ExpLocal on the seasonal component, and CV+ on the noise component:

```sh
python src/runner.py --dataset sales --basemodel Linear --cp_method_trend aci --cp_method_seasonal local_cp --cp_method_noise cv_plus --local_cp_method periodic --use_region True --use_exponential True
```
## Cite
If you find this work helpful, please cite

```bibtex
@misc{prinzhorn2024conformaltimeseriesdecomposition,
      title={Conformal time series decomposition with component-wise exchangeability}, 
      author={Derck W. E. Prinzhorn and Thijmen Nijdam and Putri A. van der Linden and Alexander Timans},
      year={2024},
      eprint={2406.16766},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2406.16766}, 
}
```

## Keywords
Time Series, Time Series Decomposition, Uncertainty, Conformal Prediction, Exchangeability Regimes, Machine Learning