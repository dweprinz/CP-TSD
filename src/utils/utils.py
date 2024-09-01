import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import wandb
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import os
import random
import torch


    
    
def plot_test_PIs(
    true,
    pred_mean,
    PI_low=None,
    PI_hi=None,
    conf_PI_low=None,
    conf_PI_hi=None,
    x_lims=None,
    scaler=None,
    label_pi=None,
    x_label=None,
    y_label=None,
    title=None,
):
    if scaler:
        true = scaler.inverse_transform_y(true)
        pred_mean = scaler.inverse_transform_y(pred_mean)
    true = true.flatten()
    pred_mean = pred_mean.flatten()

    plt.set_cmap("tab10")
    plt.cm.tab20(0)
    plt.figure(figsize=(12, 3.5))
    plt.plot(np.arange(true.shape[0]), true, label="True", color="k")
    plt.plot(pred_mean, label="Pred", color=plt.cm.tab10(1))

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)

    if conf_PI_low is not None:
        if scaler:
            conf_PI_low = scaler.inverse_transform_y(conf_PI_low)
            conf_PI_hi = scaler.inverse_transform_y(conf_PI_hi)
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
        conf_PI_hi = conf_PI_hi.flatten()
        conf_PI_low = conf_PI_low.flatten()
        plt.fill_between(
            np.arange(true.shape[0]),
            conf_PI_low,
            conf_PI_hi,
            alpha=0.3,
            label="Conformalized",
        )
        if PI_hi is not None and PI_low is not None:
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
            plt.plot(
                PI_low, label="original", color=plt.cm.tab10(0), linestyle="dashed"
            )
            plt.plot(PI_hi, color=plt.cm.tab10(0), linestyle="dashed")

    if (conf_PI_low is None) and (PI_low is not None):
        if scaler:
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)

        if label_pi is None:
            label_pi = "PI"

        if PI_low is not None:
            PI_hi = PI_hi.flatten()
            PI_low = PI_low.flatten()
            plt.fill_between(
                np.arange(true.shape[0]), PI_low, PI_hi, alpha=0.3, label=label_pi
            )

    if x_lims is not None:
        plt.xlim(x_lims)
    plt.legend(loc="upper right")
    plt.grid()

    plt.show()


def combine_upper_bounds(intervals):
    """We combine the upper bounds of the intervals to get a reconstructed conformal interval"""
    
    trend_upper = intervals[0]
    seasonality_upper = intervals[1]
    noise_upper = intervals[2]
    
    # combine the upper bounds
    combined = trend_upper + seasonality_upper + noise_upper
    
    return combined, trend_upper, seasonality_upper, noise_upper

def combine_lower_bounds(intervals):
    """We combine the lower bounds of the intervals to get a reconstructed conformal interval"""
    
    # these are coordinates of the lower bounds
    trend_lower = intervals[0]
    seasonality_lower = intervals[1]
    noise_lower = intervals[2]
    
    # combine the lower bounds
    combined = trend_lower + seasonality_lower + noise_lower
    
    return combined, trend_lower, seasonality_lower, noise_lower


def log_metrics(logger, prefix, metrics):
    logger.log({
        f"{prefix}/RMSE": metrics["RMSE"],
        f"{prefix}/PICP": metrics["PICP"],
        f"{prefix}/PIAW": metrics["PIAW"],
        f"{prefix}/PINAW": metrics["PINAW"]
    })

def log_fig(logger, folder, name, fig):
    logger.log({
        f"{folder}/{name}": wandb.Image(fig),
    })
    
def print_cp_name_and_component(cp_name, data_component):
    print(f"\n#############################################")
    print(f"Running conformal prediction for {data_component} using method {cp_name}.\n")
    

def reconstruct_interval(preds_all_comps, y_pis_all_comps):
    
    lower_bounds = [bound[:,0] for bound in y_pis_all_comps]  # all are flattened
    upper_bounds = [bound[:,1] for bound in y_pis_all_comps]  # all are flattened
    
    # combine upper bounds
    combined_upper, _, _, _ = combine_upper_bounds(upper_bounds)
    
    # combine lower bounds
    combined_lower, _, _, _ = combine_lower_bounds(lower_bounds)
    
    # sum the predictions
    summed_preds = np.sum(preds_all_comps, axis=0)
    
    return summed_preds, combined_lower, combined_upper
  
def combine_decomposed_results(results):
    # we have to combine the results of the different data component
    preds_all_comps = np.array([results[comp]["y_pred"] for comp in ["trend", "seasonal", "noise"]])
    y_pis_all_comps = np.array([results[comp]["y_pis"] for comp in ["trend", "seasonal", "noise"]])
    
    summed_preds, combined_lower, combined_upper = reconstruct_interval(preds_all_comps, y_pis_all_comps)
    return summed_preds, combined_lower, combined_upper


def _check_configuration(args):
    """Check whether the configuration has any anomalies."""

    # check dataset
    assert args.dataset in [
        "synthetic",
        "temperature",
        "natural-gas",
        "DJIA",
        "energy-consumption",
        "sales",
        "air-quality",
    ], f"Invalid dataset {args.dataset}, choose from 'synthetic', 'temperature', 'natural-gas', 'DJIA', 'energy-consumption', 'sales', 'air-quality'"

    valid_base_models = ["MLP", "Linear", "RandomForest", "GradientBoosting"]
    assert args.basemodel in valid_base_models, f"Invalid basemodel {args.basemodel}, choose from {valid_base_models}"

    valid_cp_methods = [
        "enbpi",
        "aci",
        "naive",
        "cv_plus",
        "jackknife-base",
        "jackknife_plus",
        "jackknife-minmax",
        "cqr",
        "local_cp",
    ]
    assert (
        args.cp_method_original in valid_cp_methods
    ), f"Invalid conformal method {args.cp_method_original}, choose from {valid_cp_methods}"
    assert (
        args.cp_method_trend in valid_cp_methods
    ), f"Invalid conformal method {args.cp_method_trend}, choose from {valid_cp_methods}"
    assert (
        args.cp_method_seasonal in valid_cp_methods
    ), f"Invalid conformal method {args.cp_method_seasonal}, choose from {valid_cp_methods}"
    assert (
        args.cp_method_noise in valid_cp_methods
    ), f"Invalid conformal method {args.cp_method_noise}, choose from {valid_cp_methods}"

    # when using cqr check what basemodels are used in combination with quantile models
    # this is for consistency across methods
    if args.cp_method_original == "cqr":
        valid_quantile_models = ["QuantileLinear", "GradientBoosting"]
        assert (
            args.quantilemodel in valid_quantile_models
        ), f"Invalid quantile model {args.quantilemodel}, choose from {valid_quantile_models}"
        if args.quantilemodel == "QuantileLinear":
            assert args.basemodel in [
                "Linear"
            ], "A quantile linear model can only be used with a linear model for mean regression"
        elif args.quantilemodel == "GradientBoosting":
            assert args.basemodel in [
                "GradientBoosting"
            ], "A quantile gradient boosting model can only be used with a gradient boosting or random forest model for mean regression"

    return None


def select_features(args, cp_name, data_component, time_series):
    df_train = data_component.train
    df_cal = data_component.cal
    df_test = data_component.test

    if cp_name == "cqr":
        if args.quantilemodel == "QuantileLinear":
            feature_cols = time_series.features + time_series.y_lag_features
            target_col = "y"
        elif args.quantilemodel == "GradientBoosting":
            # add 'diff' at the end of the feature names
            feature_cols = time_series.y_lag_diff_features
            target_col = "y_diff_1"

    elif cp_name in [
        "enbpi",
        "aci",
        "naive",
        "cv_plus",
        "jackknife-base",
        "jackknife_plus",
        "jackknife-minmax",
        "local_cp",
    ]:
        if args.basemodel in ["GradientBoosting", "RandomForest"]:
            feature_cols = time_series.y_lag_diff_features
            target_col = "y_diff_1"

        elif args.basemodel in ["Linear", "MLP"]:
            feature_cols = time_series.features + time_series.y_lag_features
            target_col = "y"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_cal = df_cal[feature_cols]
    y_cal = df_cal[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # make sure num features is equal to the length of the feature columns
    assert len(feature_cols) == len(
        X_train.iloc[0]
    ), "Number of features is not equal to the length of the feature columns"

    return X_train, y_train, X_cal, y_cal, X_test, y_test

def replace_infs(y_pred, y_test, y_pis, method="max"):
    upper = y_pis[:, 1]
    lower = y_pis[:, 0]

    # get all indices where there are inf or nan values
    inf_indices = np.isinf(upper) | np.isinf(lower) | np.isnan(upper) | np.isnan(lower)
    residuals = np.abs(y_test - y_pred)

    # inf gets replaced by the max value
    if method == "max":
        max_residual = np.max(residuals)
        upper[inf_indices] = y_pred[inf_indices] + max_residual
        lower[inf_indices] = y_pred[inf_indices] - max_residual

    # inf gets replaced by the median residual
    elif method == "median":
        median_residual = np.median(residuals)
        upper[inf_indices] = y_pred[inf_indices] + median_residual
        lower[inf_indices] = y_pred[inf_indices] - median_residual
    else:
        raise ValueError("Invalid method")

    return y_pis

def add_prev_y_true(data_component, y_pred, y_pis):
    y_test = data_component.test["y"].values
    first_element = y_test[0]
    y_test_prev = np.insert(y_test[:-1], 0, first_element)

    assert y_pred.shape[0] == y_test.shape[0], "y_pred and y_test must have the same length."
    assert y_pis.shape[0] == y_test.shape[0], "y_pis must have the same number of rows as y_test."
    assert y_pis.shape[1] == 2, "y_pis must have two columns."

    assert (
        y_test_prev[0] == y_test[0]
    ), "First element of y_test_prev should be the same as the first element of y_test."
    assert np.all(
        y_test_prev[1:] == y_test[:-1]
    ), "y_test_prev should be correctly shifted by one position compared to y_test."

    y_pred += y_test_prev
    y_pis[:, 0] += y_test_prev
    y_pis[:, 1] += y_test_prev

    return y_pred, y_pis

def plot_results(logger, args, data_component, lower_bounds, upper_bounds, y_pred):
    df_train = pd.concat([data_component.train, data_component.cal])
    df_test = data_component.test

    # zoomed in plot for the test set
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(df_test.index, df_test["y"], label="test")
    ax.plot(df_test.index, y_pred, label="prediction test")
    ax.fill_between(df_test.index, lower_bounds, upper_bounds, alpha=0.2, label="prediction interval")
    ax.legend()
    ax.set_title("Zoomed in plot for the test set")
    ax.set_xlabel("Time")
    ax.set_ylabel("y")
    ax.set_xlim([df_test.index[0], df_test.index[-1]])
    ax.set_ylim([df_test["y"].min(), df_test["y"].max()])
    return fig, ax

def _set_seed(seed: int):
    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # don't set seed for all possible sources of randomness
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    return None
