import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

def local_coverage(y_test, y_lower, y_upper, interval: tuple[int, int]):
    """Compute coverage for a specified interval."""
    l, r = interval
    not_covered = ~(
        (y_test[l:r] >= y_lower[l:r])
        & (y_test[l:r] <= y_upper[l:r])
    )
    return 1 - np.mean(not_covered)

def PICP(y_test, y_lower, y_upper):
    assert y_test.shape == y_lower.shape, "y_test and y_lower must have the same shape"
    return np.mean((y_lower <= y_test + 1e-9) & (y_test <= y_upper + 1e-9))

def PIAW(y_lower, y_upper):
    avg_length = np.mean(abs(y_upper - y_lower))
    return avg_length

def PINAW(y_test, y_lower, y_upper):
    avg_length = np.mean(abs(y_upper - y_lower))
    R = y_test.max() - y_test.min()
    norm_avg_length = avg_length / R
    return norm_avg_length

def CWC(y_test, y_lower, y_upper, eta: int = 30, alpha: float = 0.1):
    pinaw = PINAW(y_test, y_lower, y_upper)
    picp = PICP(y_test, y_lower, y_upper)
    return (1 - pinaw) * np.e ** (-eta * (picp - (1 - alpha)) ** 2)

def computeAllMetrics(y_test, y_pred, y_lower, y_upper):
    return {
        "RMSE": RMSE(y_test, y_pred),
        "PICP": PICP(y_test, y_lower, y_upper),
        "PIAW": PIAW(y_lower, y_upper),
        "PINAW": PINAW(y_test, y_lower, y_upper),
    }

def get_average_metrics(runs: list[dict]) -> dict:
    return {
        "RMSE": np.mean([m["RMSE"] for m in runs]),
        "PICP": np.mean([m["PICP"] for m in runs]),
        "PIAW": np.mean([m["PIAW"] for m in runs]),
        "PINAW": np.mean([m["PINAW"] for m in runs]),
    }
    
def get_std_metrics(runs: list[dict]) -> dict:
    return {
        "RMSE": np.std([m["RMSE"] for m in runs]),
        "PICP": np.std([m["PICP"] for m in runs]),
        "PIAW": np.std([m["PIAW"] for m in runs]),
        "PINAW": np.std([m["PINAW"] for m in runs]),
    }
    
def get_avg_and_std_metrics(runs: list[dict]) -> dict:
    avg = get_average_metrics(runs)
    std = get_std_metrics(runs)
    return avg, std