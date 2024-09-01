import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm
import pandas as pd
import wandb
from sklearn.model_selection import KFold


import data as datasets
from data import Decomposition, get_data

# import models
from src.models import (
    MLPReg,
    LinearReg,
    RandomForestReg,
    MeanGradientBoostingRegressor,
    QuantileRegressor,
    QuantileGradientBoostingRegressor,
    get_model,
    get_quantile_model,
)

from conformal import EnbPI, ACI, Jackknife, CQR, PeriodicCP, FeatureDistanceCP, get_cp_method


from utils.metrics import computeAllMetrics, get_avg_and_std_metrics

from utils.utils import (
    plot_test_PIs,
    print_cp_name_and_component,
    combine_upper_bounds,
    combine_lower_bounds,
    log_metrics,
    log_fig,
    print_cp_name_and_component,
    reconstruct_interval,
    combine_decomposed_results,
    _check_configuration,
    select_features,
    replace_infs,
    add_prev_y_true,
    plot_results,
    _set_seed
)

from utils.results_logger import (
    get_config_results_path,
    get_filepath_predictions,
    get_results_csv_filename,
    load_results,
    save_results,
    load_row_name,
    insert_new_result,
    update_average_metrics_file
)



import pytest

# test the data class
def test_data():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the data class
    assert data.original.train.shape[0] == 60
    assert data.original.cal.shape[0] == 20
    assert data.original.test.shape[0] == 20
    assert data.trend.train.shape[0] == 60
    assert data.trend.cal.shape[0] == 20
    assert data.trend.test.shape[0] == 20
    assert data.seasonal.train.shape[0] == 60
    assert data.seasonal.cal.shape[0] == 20
    assert data.seasonal.test.shape[0] == 20
    assert data.noise.train.shape[0] == 60
    assert data.noise.cal.shape[0] == 20
    assert data.noise.test.shape[0] == 20
    
# test the utils
def test_utils():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the utils
    assert combine_upper_bounds([np.array([1,2]), np.array([3,4]), np.array([5,6])])[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert combine_lower_bounds([np.array([1,2]), np.array([3,4]), np.array([5,6])])[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert log_metrics(wandb, "prefix", {"RMSE": 1, "PICP": 2, "PIAW": 3, "PINAW": 4}) == None
    assert log_fig(wandb, "prefix", plt.figure()) == None
    assert print_cp_name_and_component("cp_name", "data_component") == None
    assert reconstruct_interval([np.array([1,2]), np.array([3,4]), np.array([5,6])], [np.array([1,2]), np.array([3,4]), np.array([5,6])])[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert combine_decomposed_results({"trend": {"y_pred": np.array([1,2])}, "seasonal": {"y_pred": np.array([3,4])}, "noise": {"y_pred": np.array([5,6])}}).tolist() == [1, 2, 3, 4, 5, 6]
    
# test the results logger
def test_results_logger():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the results logger
    assert get_config_results_path("config") == "results/config"
    assert get_filepath_predictions("config", "method") == "results/config/predictions_method.csv"
    assert get_results_csv_filename("config") == "results/config/results.csv"
    assert load_results("config") == None
    assert save_results("config", "results") == None
    
# test the datasets
def test_datasets():
    # test the datasets
    assert datasets.get_dataset("name") == None
    assert datasets.get_datasets() == None

# test the mean regressors
def test_mean_regressors():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the mean regressors
    assert MLPReg().fit(df, df) == None
    assert LinearReg().fit(df, df) == None
    assert RandomForestReg().fit(df, df) == None
    assert GradientBoostingReg().fit(df, df) == None
    
# test the metrics
def test_metrics():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the metrics
    assert computeAllMetrics(df, df, df, df, df, df, df, df) == None
    assert get_average_metrics(df, df, df, df, df, df, df, df) == None
    
# test the mapie conformal
def test_mapie_conformal():
    # generate some data
    t = np.arange(0, 100, 1)
    y = np.sin(t)
    df = pd.DataFrame({"t": t, "y": y})
    
    # split the data
    data = Data(df, df, df, df)
    
    # test the mapie conformal
    assert EnbPI().fit(df, df) == None
    assert ACI().fit(df, df) == None
    assert Jackknife().fit(df, df) == None
