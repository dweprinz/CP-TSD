import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import RegressorMixin
from mapie.regression import MapieRegressor, MapieTimeSeriesRegressor, MapieQuantileRegressor
from mapie.subsample import BlockBootstrap
from sklearn.model_selection import KFold

from sklearn.neighbors import NearestNeighbors
from wquantiles import quantile_1D  # https://github.com/nudomarinero/wquantiles/


############################################
# MAPIE METHODS
############################################


class EnbPI:
    def __init__(self, estimator: RegressorMixin, n_bs_samples=10, length=1, alpha=0.1):
        self.estimator = estimator
        self.n_bs_samples = n_bs_samples
        self.alpha = alpha

        cv_mapietimeseries = BlockBootstrap(
            n_resamplings=n_bs_samples, length=length, overlapping=False, random_state=42
        )

        self.mapie_regressor = MapieTimeSeriesRegressor(
            estimator=self.estimator,
            method="enbpi",
            cv=cv_mapietimeseries,  # TODO make cv configurable
        )

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "OUR ASSERT: The number of samples in X and y should be the same."
        assert X.shape[0] > 0, "OUR ASSERT: The number of samples in X should be greater than 0."
        assert X.shape[1] > 0, "OUR ASSERT: The number of features in X should be greater than 0."
        assert y.shape[0] > 0, "OUR ASSERT: The number of samples in y should be greater than 0."
        self.mapie_regressor.fit(X, y)

    def predict(self, X, ensemble=False, optimize_beta=False):
        y_pred, y_pis = self.mapie_regressor.predict(
            X, alpha=self.alpha, ensemble=ensemble, optimize_beta=optimize_beta
        )

        # y_pred shape: (n_samples,)
        # y_pis shape: (n_samples, 2, alpha) as returned by mapie
        assert (
            y_pis.shape[0] == y_pred.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

        # store final predictions and prediction intervals
        self.y_pred = y_pred

        # remove the alpha dimension as it is not needed
        self.y_pis = y_pis[:, :, 0]

        assert self.y_pis.shape == (
            y_pred.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        return y_pred, self.y_pis

    def predict_with_partial_fit(
        self, X_train, y_train, X_test, y_test, step_size=1, ensemble=True, optimize_beta=False
    ):
        """This function runs an experiment in which it:
        1. updates the model with partial_fit
        2. predicts using the updated model
        3. returns predictions using the updated model
        """

        # initialize arrays for predictions and prediction intervals
        N = len(X_test)
        y_pred = np.zeros((N,))
        y_pis = np.zeros((N, 2))  # data x bounds

        # fit the model for the training dataset
        print("Fitting the model for the training dataset...")
        self.fit(X_train, y_train)

        # make initial predictions for the first step_size data points
        X_step = X_test.iloc[:step_size]

        print(f"Making predictions on the first step of test data {X_step.index.values}...")
        y_pred_step, y_pis_step = self.predict(X_step, ensemble=ensemble, optimize_beta=optimize_beta)

        assert (
            y_pis_step.shape[0] == y_pred_step.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis_step.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis_step.shape == (
            y_pred_step.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        # store
        y_pred[:step_size] = y_pred_step
        y_pis[:step_size, :] = y_pis_step

        # predict and update in steps
        print("Making predictions on the rest of the test data...\n")
        
        step_pbar = tqdm(total=len(X_test), leave=False)
        for step in range(step_size, len(X_test), step_size):
            # update the model with partial_fit
            assert (
                X_test.iloc[(step - step_size) : step, :].shape[0] == step_size
            ), "OUR ASSERT: The number of samples in the test data should be equal to step_size."

            X_step_pfit = X_test.iloc[(step - step_size) : step, :]
            y_step_pfit = y_test.iloc[(step - step_size) : step]

            self.mapie_regressor.partial_fit(X_step_pfit, y_step_pfit, ensemble=ensemble)

            assert (
                X_test.iloc[step : (step + step_size), :].shape[0] == step_size
            ), "OUR ASSERT: The number of samples in the test data should be equal to step_size."

            # make predictions for the next step_size data points
            X_step_predict = X_test.iloc[step : (step + step_size), :]

            y_pred_step, y_pis_step = self.mapie_regressor.predict(
                X_step_predict, alpha=self.alpha, ensemble=ensemble, optimize_beta=optimize_beta
            )

            assert (
                y_pis_step.shape[0] == y_pred_step.shape[0]
            ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
            assert y_pis_step.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there are two bounds."
            assert y_pis_step.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

            # when returning the prediction intervals, remove the alpha dimension as it is not needed
            y_pred_step, y_pis_step = y_pred_step, y_pis_step[:, :, 0]

            assert y_pis_step.shape == (
                y_pred_step.shape[0],
                2,
            ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

            y_pred[step : step + step_size] = y_pred_step
            y_pis[step : step + step_size, :] = y_pis_step
            
            # update the progress bar
            step_pbar.update(step_size)
            
        step_pbar.close()

        # store final predictions and prediction intervals
        self.y_pred = y_pred
        self.y_pis = y_pis

        return y_pred, self.y_pis

    def interpret_results(self):
        """This is a helper function to interpret the results of the experiment"""

        assert self.y_pis.shape == (
            self.y_pred.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"
        assert self.y_pis.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."

        lower_bounds = self.y_pis[:, 0]
        upper_bounds = self.y_pis[:, 1]

        return lower_bounds, upper_bounds


class ACI:
    def __init__(self, estimator: RegressorMixin, alpha=0.1, gamma=0.05):
        self.estimator = estimator
        self.alpha = alpha
        self.gamma = gamma

        self.mapie_regressor = MapieTimeSeriesRegressor(
            estimator=self.estimator,
            method="aci",
        )

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "OUR ASSERT: The number of samples in X and y should be the same."
        assert X.shape[0] > 0, "OUR ASSERT: The number of samples in X should be greater than 0."
        assert X.shape[1] > 0, "OUR ASSERT: The number of features in X should be greater than 0."
        assert y.shape[0] > 0, "OUR ASSERT: The number of samples in y should be greater than 0."

        self.mapie_regressor.fit(X, y)

    def predict(self, X, ensemble=False, optimize_beta=False):
        y_pred, y_pis = self.mapie_regressor.predict(
            X, alpha=self.alpha, ensemble=ensemble, optimize_beta=optimize_beta
        )

        assert (
            y_pis.shape[0] == y_pred.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

        # store final predictions and prediction intervals
        self.y_pred = y_pred
        self.y_pis = y_pis[:, :, 0]  # remove the alpha dimension as it is not needed

        assert self.y_pis.shape == (
            y_pred.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        return y_pred, self.y_pis

    def predict_with_adapt_conformal_inference(
        self, X_train, y_train, X_test, y_test, step_size=1, ensemble=True, optimize_beta=False
    ):
        """Runs an experiment that updates the model using adapt_conformal_inference and predicts using the updated model."""
        N = len(X_test)
        y_pred = np.zeros(N)
        y_pis = np.zeros((N, 2))  # data x bounds

        # fit the model for the training dataset
        print("Fitting the model for the training dataset...")
        self.fit(X_train, y_train)

        # make initial predictions for the first step_size data points
        X_step = X_test.iloc[:step_size]

        print("Making predictions on the first step of test data...")

        y_pred_step, y_pis_step = self.predict(X_step, ensemble=ensemble, optimize_beta=optimize_beta)

        assert (
            y_pis_step.shape[0] == y_pred_step.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis_step.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis_step.shape == (
            y_pred_step.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        # store
        y_pred[:step_size] = y_pred_step
        y_pis[:step_size, :] = y_pis_step

        print("Making predictions on the rest of the test data...")
        
        step_pbar = tqdm(total=len(X_test), leave=False)
        for step in range(step_size, len(X_test), step_size):
            assert (
                X_test.iloc[(step - step_size) : step, :].shape[0] == step_size
            ), "OUR ASSERT: The number of samples in the test data should be equal to step_size."

            X_step_pfit = X_test.iloc[(step - step_size) : step, :]
            y_step_pfit = y_test.iloc[(step - step_size) : step]

            # update the model with partial_fit
            self.mapie_regressor.partial_fit(X_step_pfit, y_step_pfit, ensemble=ensemble)
            self.mapie_regressor.adapt_conformal_inference(
                X_step_pfit.to_numpy(),
                y_step_pfit.to_numpy(),
                ensemble=ensemble,
                gamma=self.gamma,
                optimize_beta=optimize_beta,
            )

            assert (
                X_test.iloc[step : (step + step_size), :].shape[0] == step_size
            ), "OUR ASSERT: The number of samples in the test data should be equal to step_size."

            # make predictions for the next step_size data points
            X_step_predict = X_test.iloc[step : (step + step_size), :]
            y_pred_step, y_pis_step = self.mapie_regressor.predict(
                X_step_predict,
                alpha=self.alpha,
                ensemble=ensemble,
                optimize_beta=optimize_beta,
                allow_infinite_bounds=True,
            )

            assert (
                y_pis_step.shape[0] == y_pred_step.shape[0]
            ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
            assert y_pis_step.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there are two bounds."
            assert y_pis_step.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

            # when returning the prediction intervals, remove the alpha dimension as it is not needed
            y_pred_step, y_pis_step = y_pred_step, y_pis_step[:, :, 0]

            assert y_pis_step.shape == (
                y_pred_step.shape[0],
                2,
            ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

            y_pred[step : step + step_size] = y_pred_step
            y_pis[step : step + step_size, :] = y_pis_step
            
            # update progress bar
            step_pbar.update(step_size)
            
        step_pbar.close()

        # store final predictions and prediction intervals
        self.y_pred = y_pred
        self.y_pis = y_pis

        return y_pred, self.y_pis

    def interpret_results(self):
        """Helper function to interpret the results of the experiment."""
        lower_bounds = self.y_pis[:, 0]
        upper_bounds = self.y_pis[:, 1]

        return lower_bounds, upper_bounds


class Jackknife:
    def __init__(self, estimator: RegressorMixin, alpha=0.1, cv=-1, method: str = "plus"):
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.mapie_regressor = MapieRegressor(estimator=self.estimator, method=self.method, cv=self.cv)

    def fit(self, X, y):
        """Note that when cv is prefit, the estimator is not fitted.
        In that case mapieregressor.fit() will fit for calibration.

        All data provided in the fit method is then used for computing conformity scores only.
        At prediction time, quantiles of these conformity scores are used to provide a prediction interval with fixed width.
        The user has to take care manually that data for model fitting and conformity scores estimate are disjoint.


        Method automatically is set to base."""
        print("Fitting the model for the training dataset...")
        self.mapie_regressor.fit(X, y)
        print("Model fitted.")

    def predict(self, X):
        print("Making predictions on the test data...")
        y_pred, y_pis = self.mapie_regressor.predict(X, alpha=self.alpha)

        # y_pred shape: (n_samples,)
        # y_pis shape: (n_samples, 2, alpha)
        assert (
            y_pis.shape[0] == y_pred.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

        # store final predictions and prediction intervals
        self.y_pred = y_pred
        # remove the alpha dimension as it is not needed
        self.y_pis = y_pis[:, :, 0]

        assert self.y_pis.shape == (
            y_pred.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        return y_pred, self.y_pis

    def interpret_results(self):
        """This is a helper function to interpret the results of the experiment"""

        assert self.y_pis.shape == (
            self.y_pred.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"
        assert self.y_pis.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."

        # lower bounds
        lower_bounds = self.y_pis[:, 0]
        # upper bounds
        upper_bounds = self.y_pis[:, 1]

        return lower_bounds, upper_bounds


############################################################################################################
# Quantile Regression
############################################################################################################


class CQR:
    def __init__(self, estimator: RegressorMixin, alpha=0.1, cv=None, **kwargs):
        print("Estimator is a class to make predictions with")
        print("Kwargs are the parameters for the estimator")
        self.estimator = estimator
        self.alpha = alpha
        self.cv = cv
        self.mapie_regressor = MapieQuantileRegressor(
            estimator=self.estimator,
        )

    def fit(self, X_train, y_train, X_calib, y_calib):
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "OUR ASSERT: The number of samples in X_train and y_train should be the same."
        assert X_train.shape[0] > 0, "OUR ASSERT: The number of samples in X_train should be greater than 0."

        assert (
            X_calib.shape[0] == y_calib.shape[0]
        ), "OUR ASSERT: The number of samples in X_calib and y_calib should be the same."
        assert X_calib.shape[0] > 0, "OUR ASSERT: The number of samples in X_calib should be greater than 0."

        print(X_train.shape, y_train.shape, X_calib.shape, y_calib.shape)
        self.mapie_regressor.fit(
            X_train, y_train, X_calib=X_calib, y_calib=y_calib
        )  # note we have to specify it specifically with =

    def predict(self, X):
        # Evaluate prediction and coverage level on testing set
        X = np.array(X)
        print(X.shape)
        y_pred_cqr, y_pis_cqr = self.mapie_regressor.predict(X, alpha=self.alpha, symmetry=False)

        assert (
            y_pis_cqr.shape[0] == y_pred_cqr.shape[0]
        ), "OUR ASSERT: The first dimension of the prediction intervals should be the same as the number of samples."
        assert y_pis_cqr.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."
        assert y_pis_cqr.shape[2] == 1, "OUR ASSERT: The alpha dimension will be removed. It should be 1."

        self.y_pred_cqr = y_pred_cqr
        self.y_pis_cqr = y_pis_cqr[:, :, 0]

        assert self.y_pis_cqr.shape == (
            y_pred_cqr.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"

        return self.y_pred_cqr, self.y_pis_cqr

    def interpret_results(self):
        """This is a helper function to interpret the results of the experiment"""

        assert self.y_pis_cqr.shape == (
            self.y_pred_cqr.shape[0],
            2,
        ), "OUR ASSERT: The shape of the prediction intervals should be (n_samples, 2)"
        assert self.y_pis_cqr.shape[1] == 2, "OUR ASSERT: The second dimension should be 2 as there as two bounds."

        # lower bounds
        lower_bounds = self.y_pis_cqr[:, 0]
        # upper bounds
        upper_bounds = self.y_pis_cqr[:, 1]

        return lower_bounds, upper_bounds


############################################
# Our methods
############################################

class SplitCP:
    def __init__(self, estimator, alpha):
        self.estimator = estimator
        self.alpha = alpha
        self.y_pis = None
        self.fitted = False

        assert self.alpha > 0 and self.alpha < 1, "Alpha should be between 0 and 1"

    def fit_basemodel(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)

    def predict_basemodel(self, X):
        return pd.Series(self.estimator.predict(X), index=X.index)

    def compute_conformity_scores(self, y_pred, y_true) -> pd.Series:
        return pd.Series(np.abs(y_pred - y_true))

    def compute_quantile(self, scores, weights=None):
        n = len(scores)
        assert n > 0, f"Length of scores should be greater than 0, but is {n}"

        weights = np.array(weights)
        if weights is not None:
            assert np.all(weights >= 0), "Weights must be non-negative"
            assert len(weights) == n, "Weights must have the same length as scores"
            assert not np.isnan(weights).any(), "Weights contain NaNs"

        assert not np.isnan(scores).any(), "Scores contain NaNs"
        
        q_val = np.ceil((1 - self.alpha) * (n + 1)) / n

        # if weights are all 1, we can use the numpy quantile function
        if weights is None or np.all(weights == 1):
            q = np.quantile(scores, q_val)
        else:
            q = quantile_1D(data=scores, quantile=q_val, weights=weights)

        return q

    def partial_fit(self, index):
        y_pred = self.y_pred_test.loc[index]
        X_test = self.X_test.loc[index]
        y_test = self.y_test.loc[index]

        # add the new point as a row to the calibration set
        self.X_cal.loc[index] = X_test

        new_score = self.compute_conformity_scores(y_pred, y_test).values[0]
        self.scores.loc[index] = new_score

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.fitted = True
        self.X_train = X_train
        self.X_cal = X_cal

        self.fit_basemodel(X_train, y_train)
        y_pred_cal = self.predict_basemodel(X_cal)

        self.scores = self.compute_conformity_scores(y_pred_cal, y_cal)

    def interpret_results(self):
        if self.y_pis is None:
            raise ValueError("Model should be fitted and predicted first")

        lower, upper = self.y_pis[:, 0], self.y_pis[:, 1]
        return lower, upper

    def predict(self):
        raise NotImplementedError


class PeriodicCP(SplitCP):
    def __init__(self, estimator, alpha, use_region=False, use_exponential=False):
        super().__init__(estimator, alpha)

        self.use_region = use_region
        self.use_exponential = use_exponential
        
    def get_phase_indices_and_weights(self, phase):
        phase_indices = np.arange(phase, self.data_length, self.period_length)

        # dictionary mapping index to weight for all indices in the phase
        weights = {}

        for phase_index in phase_indices:
            # for each phase index, get the weights for the region
            for distance in range(-self.region_width, self.region_width + 1):
                if phase_index + distance in self.X_cal.index:
                    # print(f"Phase index: {phase_index}, distance: {distance}, phase: {phase}")
                    weights[phase_index + distance] = self.weights[abs(distance)]

        return weights

    def update_phase_weights(self, all_weights_per_phase, phase, index):
        # go to all the previous phases and update the weights
        for distance in range(-self.region_width, self.region_width + 1):
            # get weights for the phase that is distance phases back
            phase_weights = all_weights_per_phase[(phase + distance) % self.period_length]
            phase_weights[index] = self.weights[abs(distance)]

        return all_weights_per_phase

    def predict(self, X_test, y_test, period_length, region_width, exponential):
        assert period_length > 0, "Period should be greater than 0"
        assert period_length < len(self.X_cal), "Period should be smaller than the calibration set"
        assert self.fitted == True, "Model should be fitted first"

        # round predictions to 10 decimal places to prevent floating point errors
        self.y_pred_test = self.predict_basemodel(X_test).round(10)
        self.data_length = len(self.X_train) + len(self.X_cal) + len(X_test)

        self.X_test = X_test
        self.y_test = y_test

        self.period_length = period_length
        self.region_width = region_width
        
        if self.use_exponential:
            self.exponential = exponential
        else:
            self.exponential = 1    
        
        print(f"exponential: {self.exponential}")
        
        if not self.use_region:
            self.region_width = 0

        self.weights = self.exponential ** np.arange(0, self.region_width + 1)

        print("Weights:")
        print(self.weights)

        all_weights_per_phase = {}

        for phase in range(period_length):
            all_weights_per_phase[phase] = self.get_phase_indices_and_weights(phase)

        upper = []
        lower = []

        print("Making predictions")

        idx_pbar = tqdm(total=len(self.y_test.index), leave=False)
        for index in self.y_test.index:
            # get phase of test point
            phase = index % period_length

            # retrieve the weights and scores for this phase
            phase_indices = all_weights_per_phase[phase].keys()
            phase_weights = list(all_weights_per_phase[phase].values())

            phase_scores = self.scores.loc[phase_indices].values
            # calculate the quantile for all points in this phase
            q = self.compute_quantile(phase_scores, phase_weights)

            # calculate the prediction interval
            lower.append(self.y_pred_test[index] - q)
            upper.append(self.y_pred_test[index] + q)

            self.partial_fit(index)
            # update phase weights with the new prediction
            all_weights_per_phase = self.update_phase_weights(all_weights_per_phase, phase, index)

            idx_pbar.update(1)
            
        idx_pbar.close()
            
        lower = np.array(lower)
        upper = np.array(upper)

        self.y_pis = np.column_stack((lower, upper))
        return self.y_pred_test, self.y_pis


class FeatureDistanceCP(SplitCP):
    def __init__(self, estimator, alpha, feature_distance_metric, k_knn):
        super().__init__(estimator, alpha)
        self.k = k_knn
        self.feature_distance_metric = feature_distance_metric
        self.fitted = False

    def normalize_min_max(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return (X - self.min_X) / (self.max_X - self.min_X)

    def get_full_soft_weights(self, features):
        if self.feature_distance_metric == "euclidean":
            # calculate the euclidean distance between the test point and all calibration points
            distances = np.linalg.norm(self.normalized_X_cal.to_numpy() - features, axis=1)
        if self.feature_distance_metric == "manhattan":
            distances = np.sum(np.abs(self.normalized_X_cal.to_numpy() - features), axis=1)
        if self.feature_distance_metric == "cosine":
            distances = np.sum(self.normalized_X_cal.to_numpy() * features, axis=1) / (
                np.linalg.norm(self.normalized_X_cal.to_numpy(), axis=1) * np.linalg.norm(features)
            )
            distances = 1 - distances

        distances = distances + 1e-10  # prevent division by zero

        weights = 1 / distances

        return weights

    def compute_quantile_knn(self, test_point_features):
        # self.knn.fit(self.normalized_X_cal.to_numpy())
        # get the indices of the most similar calibration points
        neighbor_indices = self.knn.kneighbors(test_point_features, return_distance=False)[
            0
        ]  # we only put in a single point, so we only need the first index [0]
        top_k_cal_points_indices = self.X_cal.iloc[neighbor_indices].index
        conformity_scores = self.scores.loc[top_k_cal_points_indices]
        quantile = self.compute_quantile(conformity_scores)
        return quantile

    def compute_quantile_with_soft_weights(self, test_point_features):
        weights = self.get_full_soft_weights(test_point_features)
        conformity_scores = self.scores.loc[self.X_cal.index]
        assert len(conformity_scores) == len(
            weights
        ), f"Length of conformity scores ({len(conformity_scores)}) should be equal to length of weights ({len(weights)})"
        quantile = self.compute_quantile(conformity_scores, weights)
        return quantile

    def predict(self, X_test, y_test, features, method):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_test = self.predict_basemodel(X_test).round(10)

        self.min_X = np.array(self.X_cal[features].min())
        self.max_X = np.array(self.X_cal[features].max())

        # initialize the normalized calibration set
        normalized_X_cal = self.normalize_min_max(self.X_cal[features])
        self.normalized_X_cal = pd.DataFrame(normalized_X_cal, index=self.X_cal.index, columns=features)

        # fit a knn model on all calibration points
        if method == "knn":
            self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.feature_distance_metric)
            self.knn.fit(self.normalized_X_cal.to_numpy())

        lower = []
        upper = []

    
        idx_pbar = tqdm(total=len(self.y_test.index), leave=False)
        for index in self.y_test.index:
            test_point_features = self.X_test.loc[index][features]
            test_point_features = self.normalize_min_max(test_point_features)

            if method == "knn":
                quantile = self.compute_quantile_knn(test_point_features)
            elif method == "full_soft_weights":
                quantile = self.compute_quantile_with_soft_weights(test_point_features)

            lower.append(self.y_pred_test[index] - quantile)
            upper.append(self.y_pred_test[index] + quantile)
            
            idx_pbar.update(1)
            
        idx_pbar.close()

        lower = np.array(lower)
        upper = np.array(upper)

        self.y_pis = np.column_stack((lower, upper))

        return self.y_pred_test, self.y_pis

def get_cp_method(args, cp_name, basemodel, quantilemodel):
    """returns the conformal method"""
    if cp_name == "enbpi":
        cp_method = EnbPI(estimator=basemodel, n_bs_samples=args.n_bs_samples, length=args.length, alpha=args.alpha)

    elif cp_name == "aci":
        cp_method = ACI(estimator=basemodel, alpha=args.alpha, gamma=args.gamma)

    elif cp_name == "naive":
        # is not really jackknife, but just a naive method
        cp_method = Jackknife(estimator=basemodel, alpha=args.alpha, method="naive")

    elif cp_name == "cv_plus":
        cp_method = Jackknife(
            estimator=basemodel,
            alpha=args.alpha,
            method="plus",
            cv=KFold(n_splits=args.k, shuffle=True, random_state=42),
        )

    elif cp_name == "jackknife-base":
        # https://mapie.readthedocs.io/en/latest/examples_regression/2-advanced-analysis/
        # plot-coverage-width-based-criterion.html#sphx-glr-examples-regression-2-advanced-analysis-plot-coverage-width-based-criterion-py
        cp_method = Jackknife(estimator=basemodel, alpha=args.alpha, method="base", cv="prefit")

    elif cp_name == "jackknife_plus":
        cp_method = Jackknife(estimator=basemodel, alpha=args.alpha, method="plus", cv=-1)  # -1 means leave-one-out

    elif cp_name == "jackknife-minmax":
        cp_method = Jackknife(estimator=basemodel, alpha=args.alpha, method="minmax", cv=-1)  # -1 means leave-one-out
    
    elif cp_name == "local_cp":
        if args.local_cp_method == "periodic":
            cp_method = PeriodicCP(
                estimator=basemodel, alpha=args.alpha, use_region=args.use_region, use_exponential=args.use_exponential
            )

        elif args.local_cp_method == "full_soft_weights" or args.local_cp_method == "knn":
            cp_method = FeatureDistanceCP(
                estimator=basemodel,
                alpha=args.alpha,
                k_knn=args.k_knn,
                feature_distance_metric=args.feature_distance_metric,
            )

    elif cp_name == "cqr":
        assert quantilemodel is not None, "Quantile model is None, so not defined. Provide a valid quantile model."
        cp_method = CQR(estimator=quantilemodel, alpha=args.alpha, method="cqr")

    else:
        raise ValueError("Invalid conformal method")

    return cp_method