import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import wandb
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import os

from .metrics import get_avg_and_std_metrics
from .utils import log_metrics

def get_config_results_path(args, cp_name):
    """
    Builds the configuration folder path based on the dataset, basemodel, and cp_name,
    including relevant hyperparameters for the basemodel and cp method.
    """
    # start with the dataset and the basemodel
    config_folder = f"{args.dataset}/{args.basemodel}/{args.alpha}"

    # depending on the basemodel, add the relevant hyperparameters
    if "MLP" == args.basemodel:
        config_folder += f"/params={args.n_hidden}_{args.hidden_layer_sizes}_{args.epochs}_{args.batch_size}_{args.lr}"
    # add other basemodel conditions here as needed

    # add the conformal prediction method and its relevant hyperparameters
    if "enbpi" == cp_name:
        config_folder += f"/{cp_name}/n_bs_samples={args.n_bs_samples}"
    elif "aci" == cp_name:
        config_folder += f"/{cp_name}/{args.replace_inf_method}/gamma={args.gamma}"
    elif "cv_plus" == cp_name:
        config_folder += f"/{cp_name}/k={args.k}"
    elif "local_cp" == cp_name:
        
        config_folder += f"/{cp_name}/{args.local_cp_method}"
        
        if args.local_cp_method == "knn":
            config_folder += f"/distance_metric={args.feature_distance_metric}/k={args.k_knn}"
        elif args.local_cp_method == "full_soft_weights":
            config_folder += f"/distance_metric={args.feature_distance_metric}"
        elif args.local_cp_method == "periodic":
            config_folder += f"/use_region={args.use_region}/use_exponential={args.use_exponential}"
    
    # cqr works with a different basemodel
    elif "cqr" == cp_name:
        config_folder = f"{args.dataset}/{args.quantilemodel}"
        config_folder += f"/{cp_name}/alpha={args.alpha}"
    
    else:
        raise ValueError(f"Unknown saving configuration for cp_name {cp_name}.")
    
    return config_folder

       

def get_filepath_predictions(args, run, data_component, cp_name, basefolder="saved_outputs"):
    """
    Generate a file path for saving/loading predictions, incorporating
    dataset, basemodel, cp method, and hyperparameters.
    """
    # define the base folder for predictions
    basefolder = basefolder + "/predictions"
    # construct the full configuration folder path
    config_folder = get_config_results_path(args, cp_name)
    # assemble the full folder path including the data component
    full_folder_path = os.path.join(basefolder, config_folder, data_component)
    # ensure the directory exists
    os.makedirs(full_folder_path, exist_ok=True)

    # define the filename based on the seed and run
    filename = f"{args.seed + run}.csv"
    
    return os.path.join(full_folder_path, filename)

def get_results_csv_filename(args, basefolder="saved_outputs"):
    """
    Generate the file path for the results CSV, based on the args and cp_name,
    including dataset, basemodel, and cp method hyperparameters.
    """
    basefolder = basefolder + "/results"
    # construct the full configuration folder path
    config_folder = get_config_results_path(args, args.cp_method_original)
    # assemble the full folder path
    full_folder_path = os.path.join(basefolder, config_folder)
    # ensure the directory exists
    os.makedirs(full_folder_path, exist_ok=True)
    # fixed filename for results
    filename = "results.csv"
    
    return os.path.join(full_folder_path, filename)

def load_results(filepath):
    if os.path.exists(filepath):
        print(f"Loading existing results from {filepath}")
        df_results = pd.read_csv(filepath)
        y_pred = df_results['y_pred'].values
        lower_bounds = df_results['y_lower'].values
        upper_bounds = df_results['y_upper'].values
        y_pis = np.column_stack((lower_bounds, upper_bounds))
        return y_pred, y_pis, lower_bounds, upper_bounds
    else:
        return None, None, None, None
    
    
def save_results(filepath, y_pred, lower_bounds, upper_bounds):
    """ Save the results to a CSV file.
    
    Args:
    filepath: str, the path to the CSV file
    y_pred: np.ndarray, the predicted values
    lower_bounds: np.ndarray, the lower bounds of the prediction intervals
    upper_bounds: np.ndarray, the upper bounds of the prediction intervals
    
    Returns:
    None
    """
    df_results = pd.DataFrame({
        'y_pred': y_pred,
        'y_lower': lower_bounds,
        'y_upper': upper_bounds
    })
    df_results.to_csv(filepath, index=False)
    print(f"Saved results to {filepath}")
    
    
    

def load_row_name(args, original=False):
    """Load the row name for the results CSV file."""
    if original:
        # get the row name based on the cp method for the original data component
        return getattr(args, f"cp_method_original")
    # get the row name based on the cp methods for the trend, seasonal, and noise components
    return "_".join(getattr(args, f"cp_method_{data_component}") for data_component in ["trend", "seasonal", "noise"])


def insert_new_result(args, average_metrics, row_name):
    """Insert a new row into the results CSV file.

    Args:
    args: argparse.Namespace, the parsed arguments
    average_metrics: dict, the new average metrics
    row_name: str, the name of the row to be added to the file

    Returns:
    None
    """

    # get the filename
    csv_filename = get_results_csv_filename(args, basefolder=args.save_dir)

    # load the existing file or create a new one
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=["method"] + list(average_metrics.keys()))

    # check if the row already exists
    if row_name in df["method"].values:
        print(f"Row '{row_name}' already exists in {csv_filename}. Skipping addition of new row.")
    # if it doesn't exist, add it
    else:
        new_row = pd.DataFrame([average_metrics], index=[row_name])
        new_row.insert(0, "method", row_name)  # insert the row_name as the first column

        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(csv_filename, index=False)
        print(f"Updated/created {csv_filename} with new metrics.")


def update_average_metrics_file(args, logger, all_original_metrics, all_combined_metrics):
    avg_original, std_original = get_avg_and_std_metrics(all_original_metrics)

    avg_combined, std_combined = get_avg_and_std_metrics(all_combined_metrics)

    # log the averaged metrics
    if not args.no_log:
        log_metrics(logger, f"{args.dataset}/metrics/original_avg", avg_original)
    if not args.no_log:
        log_metrics(logger, f"{args.dataset}/metrics/combined_avg", avg_combined)
    if not args.no_log:
        log_metrics(logger, f"{args.dataset}/metrics/original_std", std_original)
    if not args.no_log:
        log_metrics(logger, f"{args.dataset}/metrics/combined_std", std_combined)

    # get the rownames
    rowname_original = load_row_name(args, original=True)
    rowname_combined = load_row_name(args, original=False)

    # update the average metrics file
    insert_new_result(args, avg_combined, rowname_combined)
    insert_new_result(args, avg_original, rowname_original)

