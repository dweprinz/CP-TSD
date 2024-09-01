from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor

# MLPReg, LinearReg, RandomForestReg, GradientBoostingReg
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# https://pypi.org/project/quantnn/
# !pip install quantnn

# https://pypi.org/project/quantile-forest/
# !pip install quantile-forest

##################
# Quantile regressors
##################

class QuantileRegressor(QuantileRegressor):
    def __init__(self, quantile: float = 0.5, alpha: int = 0, solver: str = "highs"):
        """_summary_

        Args:
            quantile (float, optional): Quantile to predict. Defaults to 0.5.
            alpha (int, optional): Regularization constant for L1. Defaults to 0.
            solver (str, optional): Linear programming formulation.. Defaults to "highs".
        """
        super().__init__(quantile=quantile, alpha=alpha, solver=solver)


class QuantileGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self, loss: str = "quantile", alpha: float = 0.5):
        super().__init__(loss=loss, alpha=alpha)

def get_quantile_model(args):
    if args.quantilemodel == "QuantileLinear":
        model = QuantileRegressor()
    elif args.quantilemodel == "GradientBoosting":
        model = QuantileGradientBoostingRegressor()

    elif args.quantilemodel is None:
        model = None

    else:
        raise ValueError("Invalid quantile model")

    return model


##################
# Mean regressors
##################



class MLPReg(MLPRegressor):
    def __init__(
        self,
        hidden_layer_sizes=(100, 100),
        max_iter=100,
        learning_rate_init=1e-3,
        batch_size=32,
        early_stopping=True,
        verbose=True,
    ):
        assert learning_rate_init > 0, "Learning rate should be greater than 0"
        assert learning_rate_init < 1, "Learning rate should be less than 1"

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    # Additional methods specific to MLPReg can be added here.


class LinearReg(LinearRegression):
    def __init__(self):
        super().__init__()

    # Additional methods specific to LinearReg can be added here.


class RandomForestReg(RandomForestRegressor):
    def __init__(self):
        super().__init__()

    # Additional methods specific to RandomForestReg can be added here.


class MeanGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self):
        super().__init__()

    # Additional methods specific to GradientBoostingReg can be added here.



def get_model(args):
    """returns the model"""
    if args.basemodel == "MLP":
        sizes = tuple([args.hidden_layer_sizes for _ in range(args.n_hidden)])
        model = MLPReg(
            hidden_layer_sizes=sizes,
            max_iter=args.epochs,
            learning_rate_init=args.lr,
            batch_size=args.batch_size,
            verbose=False,
        )

    elif args.basemodel == "Linear":
        model = LinearReg()

    elif args.basemodel == "RandomForest":
        model = RandomForestReg()

    elif args.basemodel == "GradientBoosting":
        model = MeanGradientBoostingRegressor()
    else:
        raise ValueError("Invalid basemodel")

    return model