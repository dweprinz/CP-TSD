### LIBRARIES
import pandas as pd

import argparse
import wandb

### MODULES

# import data modules
from data import get_data

# import models
from models import (
    get_model,
    get_quantile_model,
)

from conformal import get_cp_method

# import utils
from utils.metrics import computeAllMetrics

from utils.utils import (
    print_cp_name_and_component,
    log_metrics,
    log_fig,
    print_cp_name_and_component,
    combine_decomposed_results,
    _check_configuration,
    select_features,
    replace_infs,
    add_prev_y_true,
    plot_results,
    _set_seed
)
from utils.results_logger import (
    get_filepath_predictions,
    load_results,
    save_results,
    update_average_metrics_file
)

def run_cp_model(logger, args, time_series, cp_name, data_component, model, cp_method):
    # select correct features (based on the basemodel)
    X_train, y_train, X_cal, y_cal, X_test, y_test = select_features(args, cp_name, data_component, time_series)

    # initialize y_pred and y_pis
    y_pred, y_pis = None, None
    
    def add_y_true_if_needed(y_pred, y_pis):
        if args.basemodel in ["GradientBoosting", "RandomForest"]:
            return add_prev_y_true(data_component, y_pred, y_pis)
        return y_pred, y_pis
    
    # run experiment based on cp_name
    if cp_name == "enbpi":
        y_pred, y_pis = cp_method.predict_with_partial_fit(
            X_train=pd.concat([X_train, X_cal]),
            y_train=pd.concat([y_train, y_cal]),
            X_test=X_test,
            y_test=y_test,
            step_size=1,
            ensemble=True,
            optimize_beta=False,
        )

        y_pred, y_pis = add_y_true_if_needed(y_pred, y_pis)
    elif cp_name == "aci":
        y_pred, y_pis = cp_method.predict_with_adapt_conformal_inference(
            X_train=pd.concat([X_train, X_cal]),
            y_train=pd.concat([y_train, y_cal]),
            X_test=X_test,
            y_test=y_test,
            step_size=1,
        )

        y_pred, y_pis = add_y_true_if_needed(y_pred, y_pis)
        
    elif cp_name in ["jackknife_plus", "cv_plus"]:
        cp_method.fit(X=pd.concat([X_train, X_cal]), y=pd.concat([y_train, y_cal]))
        y_pred, y_pis = cp_method.predict(X=X_test)
        y_pred, y_pis = add_y_true_if_needed(y_pred, y_pis)
        
    elif cp_name == "local_cp":
        cp_method.fit(X_train, y_train, X_cal, y_cal)

        if args.local_cp_method == "periodic":
            
            if args.use_exponential:
                region_width = time_series.region_width_exp
            else:
                region_width = time_series.region_width_local
                
            y_pred, y_pis = cp_method.predict(X_test, y_test, time_series.period_length, region_width, time_series.exponential)
        
        elif args.local_cp_method in ["knn", "full_soft_weights"]:
            y_pred, y_pis = cp_method.predict(
                X_test, y_test, features=time_series.distance_features, method=args.local_cp_method
            )
        
        y_pred, y_pis = add_y_true_if_needed(y_pred, y_pis)
            
    elif cp_name == "cqr":
        cp_method.fit(X_train, y_train, X_cal, y_cal)
        y_pred, y_pis = cp_method.predict(X_test)

        if args.quantilemodel == "GradientBoosting":
            y_pred, y_pis = add_prev_y_true(data_component, y_pred, y_pis)

    if args.replace_inf_method is not None:
        # replace possible inf values in the prediction intervals
        y_pis = replace_infs(y_pred, y_test, y_pis, method=args.replace_inf_method)

    return y_pred, y_pis


def main(logger, args):
    time_series, data, fig, ax = get_data(args)
    _check_configuration(args)

    if not args.no_log:
        log_fig(logger, folder=f"{args.dataset}/figs", name="decomposition", fig=fig)

    all_original_metrics = []
    all_combined_metrics = []

    for run in range(args.n_runs):
        print(f"\nRun {run + 1}/{args.n_runs}")
        seed = args.seed + run
        _set_seed(seed)  # adjust seed for each run

        data_components_results = {}

        for data_component in ["original", "trend", "seasonal", "noise"]:
            args_cp_name = f"cp_method_{data_component}"
            cp_name = getattr(args, args_cp_name)
            data_comp = getattr(data, data_component)

            model = get_model(args)
            quantilemodel = get_quantile_model(args)

            # load or run the conformal prediction method
            filepath = get_filepath_predictions(args, run, data_component, cp_name, basefolder=args.save_dir)
            y_pred, y_pis, lo_bounds, up_bounds = load_results(filepath)

            if y_pred is None or y_pis is None:
                print_cp_name_and_component(cp_name, data_component)

                # get the conformal prediction method
                cp_method = get_cp_method(args, cp_name, model, quantilemodel)

                # run the conformal prediction method
                y_pred, y_pis = run_cp_model(logger, args, time_series, cp_name, data_comp, model, cp_method)
                
                # save the results
                lo_bounds, up_bounds = cp_method.interpret_results()
                save_results(filepath, y_pred, lo_bounds, up_bounds)
            else:
                # if the results are loaded, print a message
                print(f"Results for '{data_component}' loaded successfully.\n")

            # plot the results and calculate the metrics
            fig, _ = plot_results(logger, args, data_comp, lo_bounds, up_bounds, y_pred)
            if not args.no_log:
                    log_fig(logger, folder=f"{args.dataset}/figs", name=f"{data_component}_seed_{seed}", fig=fig)

            # compute metrics for the current data component
            component_metrics = computeAllMetrics(
                y_test=data_comp.test["y"].values, y_pred=y_pred, y_lower=lo_bounds, y_upper=up_bounds
            )

            if data_component == "original":
                all_original_metrics.append(component_metrics)
            else:
                data_components_results[data_component] = {"y_pred": y_pred,"y_pis": y_pis}
                if not args.no_log:
                    log_metrics(logger, f"{args.dataset}/metrics/{data_component}", component_metrics)


        # combine the results of the different data components
        sum_preds, comb_lo, comb_up = combine_decomposed_results(data_components_results)

        # plot the combined results
        fig, ax = plot_results(logger, args, data.original, comb_lo, comb_up, sum_preds)

        # log the combined figure
        if not args.no_log:
            log_fig(logger, folder=f"{args.dataset}/figs", name=f"combined_seed_{seed}", fig=fig)

        # we calculate the metrics for the original as well as the combined results
        y_test = data.original.test["y"].values
        comb_metrics = computeAllMetrics(y_test=y_test, y_pred=sum_preds, y_lower=comb_lo, y_upper=comb_up)
        all_combined_metrics.append(comb_metrics)

    ########################################
    # update the average metrics file
    ########################################
    update_average_metrics_file(args, logger, all_original_metrics, all_combined_metrics)


def parse_args():
    """https://docs.wandb.ai/ref/python/log"""
    parser = argparse.ArgumentParser(
        prog="Conformal Prediction for TS", description="Run conformal methods on a dataset"
    )

    # seed
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--n_runs", type=int, required=False, default=3)
    parser.add_argument("--no_log", action="store_true", help="Disable logging")

    # data
    parser.add_argument("--dataset", type=str, required=True, default=None)

    # basemodels
    parser.add_argument("--basemodel", type=str, required=False, default="Linear", choices=["Linear", "MLP", "GradientBoosting"])
    parser.add_argument("--n_hidden", type=int, required=False, default=2)
    parser.add_argument("--hidden_layer_sizes", type=int, required=False, default=100)
    parser.add_argument("--epochs", type=int, required=False, default=100)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch_size", type=int, required=False, default=32)

    parser.add_argument("--quantilemodel", type=str, required=False, default=None)

    # conformal methods
    ########################################
    parser.add_argument("--cp_method_original", type=str, required=False, default="enbpi", choices=["enbpi", "aci", "cv_plus", "local_cp"])
    parser.add_argument("--cp_method_trend", type=str, required=False, default="enbpi", choices=["enbpi", "aci", "cv_plus", "local_cp"])
    parser.add_argument("--cp_method_seasonal", type=str, required=False, default="enbpi", choices=["enbpi", "aci", "cv_plus", "local_cp"])
    parser.add_argument("--cp_method_noise", type=str, required=False, default="enbpi", choices=["enbpi", "aci", "cv_plus", "local_cp"])

    parser.add_argument("--alpha", type=float, required=False, default=0.1)

    # parameters for ACI method
    parser.add_argument("--gamma", type=float, required=False, default=0.01)
    parser.add_argument("--replace_inf_method", type=str, required=False, default="max")

    # parameters for EnbPI or EnCQR method
    parser.add_argument("--n_bs_samples", type=int, required=False, default=20)
    parser.add_argument("--length", type=int, required=False, default=1)

    # parameters for Jackknife method
    parser.add_argument("--k", type=int, required=False, default=20)
    ########################################

    parser.add_argument("--local_cp_method", type=str, required=False, default=None, choices=["periodic", "knn", "full_soft_weights"])
    
    # when using local_cp_method = "periodic"
    parser.add_argument("--use_region", type=bool, required=False, default=False)
    parser.add_argument("--use_exponential", type=bool, required=False, default=False)
    
    parser.add_argument("--k_knn", type=int, required=False, default=20) # when using local_cp_method = "knn"
    parser.add_argument("--feature_distance_metric", type=str, required=False, default="euclidean") # when using local_cp_method = "knn" or "full_soft_weights"

    # saving
    parser.add_argument(
        "--save_dir", type=str, required=False, default="./saved-outputs"
    ) 

    return parser.parse_args()

# examplary command: python runner.py --no_log --seed 42 --n_runs 3 --dataset synthetic --basemodel MLP --n_hidden 2 --hidden_layer_sizes 100 --epochs 100 --lr 1e-3 --batch_size 32 --cp_method_original enbpi --cp_method_trend enbpi --cp_method_seasonal enbpi --cp_method_noise enbpi --alpha 0.1 --n_bs_samples 20 --length 1 --k 10

if __name__ == "__main__":
    args = parse_args()
    logging_enabled = not args.no_log

    logger = wandb.init(project="project_name", config=args) if logging_enabled else None
    main(logger, args)
