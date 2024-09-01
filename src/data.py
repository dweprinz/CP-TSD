import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import warnings

from dataclasses import dataclass


@dataclass
class DataSplit:
    train: pd.DataFrame
    cal: pd.DataFrame
    test: pd.DataFrame

    def __init__(self, df: pd.DataFrame, sizes=[0.6, 0.2, 0.2]):
        self.train = df.iloc[: int(sizes[0] * len(df))]
        self.cal = df.iloc[int(sizes[0] * len(df)) : int((sizes[0] + sizes[1]) * len(df))]
        self.test = df.iloc[int((sizes[0] + sizes[1]) * len(df)) :]


@dataclass
class Data:
    original: DataSplit
    trend: DataSplit
    seasonal: DataSplit
    noise: DataSplit

    def __init__(
        self,
        original: pd.DataFrame,
        trend: pd.DataFrame,
        seasonal: pd.DataFrame,
        noise: pd.DataFrame,
        sizes=[0.6, 0.2, 0.2],
    ):
        self.original = DataSplit(original, sizes=sizes)
        self.trend = DataSplit(trend, sizes=sizes)
        self.seasonal = DataSplit(seasonal, sizes=sizes)
        self.noise = DataSplit(noise, sizes=sizes)

class Decomposition:
    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.t = t
        self.y = y
        
    def time_series_to_df(self):
        df = pd.DataFrame({"t": self.t, "y": self.y})
        df = df.set_index("t")
        return df
    
    def decompose_stl(self, period="monthly"):
        if period == "weekly":
            season = 7
        elif period == "monthly":
            season = 30
        elif period == "yearly":
            season = 365

        # decompose the time series
        decomposition = STL(self.time_series_to_df(), period=season)
        decomposition = decomposition.fit()

        return decomposition
    
    def decompose_seasonal(self, period="monthly"):
        if period == "weekly":
            season = 7
        elif period == "monthly":
            season = 30
        elif period == "yearly":
            season = 365

        decomposition = seasonal_decompose(self.y, period=season)

        return decomposition
        
    def plot_decomposition(self, method="stl"):
        if method == "stl":
            return self.decompose_stl().plot()
            
        elif method == "seasonal":
            return self.decompose_seasonal().plot()
        
        raise ValueError("Please enter a valid decomposition method")
    

############################################################################################################
# Synthetic datasets
# Synthetic datasets have synthetic decomposition components
############################################################################################################
class Synthetic(object):
    def __init__(self, root: str = "./datasets", length: int = 15 * 365) -> None:
        self.root = root
        self.pathname = str()
        self.t = np.arange(length)
        df = pd.DataFrame({"t": self.t, "y": np.nan})

        self.length = length

        self.original = df.copy()
        self.trend = df.copy()
        self.seasonality = df.copy()
        self.noise = df.copy()

        self.num_lags = 1
        self.y_lag_features = [f"y_lag_{i}" for i in range(1, self.num_lags + 1)]
        self.features = []

        self.y_lag_diff_features = [f"{feature}_diff" for feature in self.y_lag_features]
        self.diff_features = [f"{feature}_diff" for feature in self.features]

        self.nfeatures = len(self.features) + self.num_lags

        self.period_length = 30
        
        self.region_width_local = 3
        self.region_width_exp = 14
        self.exponential = 0.75

    def create_trend(self, intercept, slope):
        self.trend["y"] = intercept + slope * self.trend["t"]

    def create_seasonality(self, amplitude, periods=["monthly"]):
        # check whether t is a np.ndarray
        assert isinstance(self.t, np.ndarray), "Please create the 't' axis first."
        seasonality_values = np.zeros(self.length)

        # check whether t is a series and not a single digit
        assert len(self.t) > 1, "Please create the 't' axis first."

        for period in periods:
            if period == "daily":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 1)
            elif period == "weekly":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 7)
            elif period == "monthly":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 30)
            elif period == "quarterly":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 90)
            elif period == "trimester":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 120)
            elif period == "semi-annually":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 180)
            elif period == "yearly":
                seasonality_values += amplitude * np.sin(2 * np.pi * self.t / 365)
            else:
                print("Invalid period specified.")
                return
        self.seasonality["y"] = seasonality_values

    def create_normal_noise(self, mean, std):
        # NOTE: SEED 0 HERE
        np.random.seed(0)
        self.noise["y"] = np.random.normal(mean, std, self.length)

    def create_time_series(self, intercept=0, slope=0, amplitude=0, periods=["monthly"], noise_mean=0, noise_std=0):
        # Create each component
        self.create_trend(intercept, slope)
        self.create_seasonality(amplitude, periods)
        self.create_normal_noise(noise_mean, noise_std)

        # Combine components
        self.original["y"] = self.trend["y"] + self.seasonality["y"] + self.noise["y"]

    def create_lags_y(self):
        warnings.filterwarnings("ignore")
        components = [self.original, self.trend, self.seasonality, self.noise]

        for df in components:
            for i in range(1, self.num_lags + 1):
                df[f"y_lag_{i}"] = df["y"].shift(i)
                df[f"y_diff_{i}"] = df["y"].diff(i)
                df[f"y_lag_{i}_diff"] = df[f"y_diff_{i}"].shift(1)

            # drop NA values due to lagging and differencing
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

    def plot_pattern(self, component="original", title="Time Series", xlabel="Time", ylabel="Value"):
        plt.figure(figsize=(12, 4))

        # Check the component to plot
        if component == "original":
            pattern = self.original["y"]
        elif component == "trend":
            pattern = self.trend["y"]
        elif component == "seasonality":
            pattern = self.seasonality["y"]
        elif component == "noise":
            pattern = self.noise["y"]
        else:
            print(f"Unknown component: {component}")
            return

        plt.plot(self.original["t"], pattern)  # Ensure 't' is consistent across components
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_decomposition(self, show_plot=False):
        fig, ax = plt.subplots(4, 1, figsize=(12, 8))
        ax[0].plot(self.original["t"], self.original["y"], label="Original")
        ax[1].plot(self.trend["t"], self.trend["y"], label="Trend")
        ax[2].plot(self.seasonality["t"], self.seasonality["y"], label="Seasonal")
        ax[3].plot(self.noise["t"], self.noise["y"], label="Noise")

        if show_plot:
            plt.show()

        return fig, ax


############################################################################################################
# Real-life datasets
# Real-life datasets have real-life decomposition components using STL decomposition
############################################################################################################
class Decompositions:
    def __init__(self, t: np.ndarray, y: np.ndarray, frequency="daily"):
        self.t = t
        self.y = y

        self.frequency = frequency
        # set the original data with y
        self.original = pd.DataFrame({"t": self.t, "y": self.y})

        # set the trend, seasonality, and noise components
        df = pd.DataFrame({"t": self.t, "y": np.nan})
        self.trend = df.copy()
        self.seasonal = df.copy()
        self.noise = df.copy()

    def decompose(self, series, model="additive", seasonality="monthly"):
        # choose period

        if self.frequency == "daily":
            periods = {
                "daily": 1,
                "weekly": 7,
                "biweekly": 14,
                "monthly": 30,
                "quarterly": 90,
                "trimester": 120,
                "semi-annually": 180,
                "yearly": 365,
            }

        elif self.frequency == "hourly":
            periods = {
                "half_daily": 12,
                "daily": 24,
                "weekly": 24 * 7,
                "biweekly": 24 * 14,
                "monthly": 24 * 30,
                "quarterly": 24 * 90,
                "trimester": 24 * 120,
                "semi-annually": 24 * 180,
                "yearly": 24 * 365,
            }
        else:
            raise ValueError("Invalid frequency specified.")

        period = periods[seasonality]

        if model == "additive":
            decomposition = sm.tsa.STL(series, period=period).fit()
        elif model == "multi":
            decomposition = sm.tsa.MSTL(series, periods=list(periods.values())).fit()
        elif model == "multiplicative":
            raise NotImplementedError("Multiplicative STL decomposition not yet implemented.")
        elif model == "oneshot-stl":
            raise NotImplementedError("One-shot STL decomposition not yet implemented.")
        elif model == "online-stl":
            raise NotImplementedError("Online STL decomposition not yet implemented.")
        else:
            raise ValueError("Invalid decomposition model specified.")
            return

        self.trend["y"] = decomposition.trend
        self.seasonal["y"] = decomposition.seasonal
        self.noise["y"] = decomposition.resid

        # check whether decompositions have the same length
        assert (
            len(self.trend["y"]) == len(self.seasonal["y"]) == len(self.noise["y"])
        ), "Decompositions have different lengths."
        assert len(self.trend["y"]) == len(self.original["y"]), "Decompositions have different lengths than original."

        # change all t to integer
        self.original["t"] = np.arange(len(self.original["y"]), dtype=np.int64)
        self.trend["t"] = np.arange(len(self.trend["y"]), dtype=np.int64)
        self.seasonal["t"] = np.arange(len(self.seasonal["y"]), dtype=np.int64)
        self.noise["t"] = np.arange(len(self.noise["y"]), dtype=np.int64)

        return self.original, self.trend, self.seasonal, self.noise


class RealLifeTS:
    def __init__(self):
        self.t = None
        self.y = None
        self.data = None
        self.original = None
        self.trend = None
        self.seasonality = None
        self.noise = None

    def load(self):
        raise NotImplementedError("Load method not implemented.")

    def create_time_series(self, seasonality="monthly", frequency="daily"):
        # load the data
        data = self.load()
        self.data = data
        self.t = data.t  # here t is datetime

        # check whether t is datetime for the decomposition
        assert isinstance(self.t, pd.Series), "Time series 't' is not a pandas series."
        assert isinstance(self.t[0], pd.Timestamp), "Time series 't' is not a datetime series."
        self.y = data.y

        # decompose the data
        decomposition = Decompositions(self.t, self.y, frequency)
        self.original, self.trend, self.seasonality, self.noise = decomposition.decompose(
            self.y, seasonality=seasonality
        )

        # check whether t now is integer and series
        for i, df in enumerate([self.original, self.trend, self.seasonality, self.noise]):
            # check whether t is integer
            assert isinstance(df["t"], pd.Series), f"Time series 't' is not a pandas series. For dataframe {i+1}/4"
            assert isinstance(df["t"][0], np.int64), f"Time series 't' is not an integer array. For dataframe {i+1}/4"

            # check whether is not all nan
            assert not df["y"].isnull().all(), f"Data is all nan. For dataframe {i+1}/4"

        # get features so we can add them to each component
        features = self.data[self.features]

        # assert features and original have the same length
        assert (
            len(features) == len(self.trend) == len(self.seasonality) == len(self.noise)
        ), "Features and decompositions have different lengths."

        self.original = pd.concat([self.original, features], axis=1)
        self.trend = pd.concat([self.trend, features], axis=1)
        self.seasonality = pd.concat([self.seasonality, features], axis=1)
        self.noise = pd.concat([self.noise, features], axis=1)

        return self.data

    def create_lags_y(self):
        warnings.filterwarnings("ignore")
        components = [self.original, self.trend, self.seasonality, self.noise]

        for df in components:
            for i in range(1, self.num_lags + 1):
                df[f"y_lag_{i}"] = df["y"].shift(i)
                df[f"y_diff_{i}"] = df["y"].diff(i)
                df[f"y_lag_{i}_diff"] = df[f"y_diff_{i}"].shift(1)

            # drop NA values due to lagging and differencing
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

        # check that data is not all nan
        for i, df in enumerate([self.original, self.trend, self.seasonality, self.noise]):
            assert not df["y"].isnull().all(), f"Data is all nan. For dataframe {i+1}/4"

    def create_diff_features(self):
        components = [self.original, self.trend, self.seasonality, self.noise]

        for df in components:
            for feature in self.features:
                df[f"{feature}_diff"] = df[feature].diff()

    def plot_pattern(self, show_plot=False, component="original", title="Time Series", xlabel="Time", ylabel="Value"):
        fig, ax = plt.subplots(figsize=(12, 4))

        # Check the component to plot
        if component == "original":
            pattern = self.original["y"]
        elif component == "trend":
            pattern = self.trend["y"]
        elif component == "seasonality":
            pattern = self.seasonality["y"]
        elif component == "noise":
            pattern = self.noise["y"]
        else:
            print(f"Unknown component: {component}")
            return

        ax.plot(self.original["t"], pattern)  # ensure 't' is consistent across components

        if show_plot:
            plt.show()

        return fig, ax

    def plot_decomposition(self, show_plot=False):
        fig, ax = plt.subplots(4, 1, figsize=(12, 8))
        ax[0].plot(self.original["t"], self.original["y"], label="Original")
        ax[1].plot(self.trend["t"], self.trend["y"], label="Trend")
        ax[2].plot(self.seasonality["t"], self.seasonality["y"], label="Seasonal")
        ax[3].plot(self.noise["t"], self.noise["y"], label="Noise")

        if show_plot:
            plt.show()

        return fig, ax


class TemperatureTS(RealLifeTS):
    def __init__(self, root="./datasets/temperature") -> None:
        super().__init__()
        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.num_lags = 1

        self.y_lag_features = [f"y_lag_{i}" for i in range(1, self.num_lags + 1)]
        # self.features = ["meanpressure", "humidity", "wind_speed"]
        self.features = []

        self.y_lag_diff_features = [f"{feature}_diff" for feature in self.y_lag_features]
        self.diff_features = [f"{feature}_diff" for feature in self.features]

        self.nfeatures = len(self.features) + self.num_lags

        self.period_length = 365
        self.region_width = 30

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a csv file

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        train_data = pd.read_csv(self.root + "/DailyDelhiClimateTrain.csv")

        # remove 2017-01-01,10.0,100.0,0.0,1016.0 from train
        train_data = train_data[train_data["date"] != "2017-01-01"]

        test_data = pd.read_csv(self.root + "/DailyDelhiClimateTest.csv")

        # concatenate the train and test data
        data = pd.concat([train_data, test_data], axis=0)

        # get the date (datetime) as index of series
        data.index = data["date"].apply(pd.to_datetime)
        data = data.drop(columns=["date"])

        # sort the series by date
        data = data.sort_index()

        # set column names correctly
        data["t"] = data.index
        data["y"] = data["meantemp"]

        # interpolate rows with meanpressure higher than 1100 and lower than 900 (clear outliers)

        outlier_rows_idx = data[(data["meanpressure"] > 1100) | (data["meanpressure"] < 900)].index
        data.loc[outlier_rows_idx, "meanpressure"] = np.nan
        data["meanpressure"] = data["meanpressure"].interpolate()

        # print num of outliers interpolated
        print(f"Number of outliers interpolated: {len(outlier_rows_idx)}")

        # only use the t and y columns
        data = data.reset_index(drop=True)
        self.data = data

        return self.data


class NaturalGasTS(RealLifeTS):
    def __init__(self, root="./datasets/natural-gas") -> None:
        super().__init__()
        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.features = ["y_lag_1"]
        self.nfeatures = len(self.features)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a xls file

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        # extracting only the columns corresponding to time (t) and value (y) into a new dataframe
        # skipping the first two rows which are headers and metadata

        sheet_names = pd.ExcelFile(self.root + "/RNGWHHDm.xls").sheet_names  # get all sheet names

        data_sheet = pd.read_excel(self.root + "/RNGWHHDm.xls", sheet_name=sheet_names[1])

        data_cleaned = data_sheet.iloc[2:].copy()  # copy starting from the third row to skip metadata
        data_cleaned.columns = ["t", "y"]  # rename columns for clarity

        # converting 't' to datetime format and 'y' to float for proper data handling
        data_cleaned["t"] = pd.to_datetime(data_cleaned["t"])
        data_cleaned["y"] = pd.to_numeric(data_cleaned["y"], errors="coerce")

        data_cleaned.reset_index(drop=True, inplace=True)  # reset index for the new dataframe

        data = data_cleaned

        # only use the t and y columns
        data = data[["t", "y"]].reset_index(drop=True)
        self.data = data

        return self.data


class DJIATS(RealLifeTS):
    def __init__(self, root="./datasets/stock") -> None:
        super().__init__()
        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.features = ["y_lag_1"]
        self.nfeatures = len(self.features)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a csv file

        https://www.wsj.com/market-data/quotes/index/DJIA/historical-prices

        from 01/01/1900 to 18/02/2024

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        data = pd.read_csv(self.root + "/DJIA.csv")

        # get the date (datetime) as index of series
        data.index = data["Date"].apply(pd.to_datetime)

        # strip all column names from white spaces
        data.columns = data.columns.str.strip()

        # sort the series by date
        data = data.sort_index()

        # set column names correctly
        data["t"] = data.index

        data["y"] = data["Close"]

        # only use the t and y columns
        data = data[["t", "y"]].reset_index(drop=True)
        self.data = data

        return self.data


class EnergyConsumption(RealLifeTS):
    def __init__(self, root="./datasets/energy_consumption") -> None:
        super().__init__()

        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.num_lags = 5
        self.y_lag_features = [f"y_lag_{i}" for i in range(1, self.num_lags + 1)]
        self.features = [
            "hour",
            "day",
            "month",
            "year",
            "weekday",
            "season",
            "holiday",
            "non_working",
            "DailyCoolingDegreeDays",
            "DailyHeatingDegreeDays",
            "HourlyDryBulbTemperature",
            "AC_kW",
        ]

        self.y_lag_diff_features = [f"{feature}_diff" for feature in self.y_lag_features]
        self.diff_features = [f"{feature}_diff" for feature in self.features]

        self.nfeatures = len(self.features) + self.num_lags

        self.period_length = 12
        
        self.region_width_local = 2
        self.region_width_exp = 5
        self.exponential = 0.5

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a csv file

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        data = pd.read_csv(self.root + "/hourly1418_energy_temp_PV.csv")

        # cut to 60% of the data
        data = data.iloc[: int(0.6 * len(data))]

        # get the date (datetime) as index of series
        data.index = data["Dates"].apply(pd.to_datetime)

        # strip all column names from white spaces
        data.columns = data.columns.str.strip()

        # sort the series by date
        data = data.sort_index()

        # set column names correctly
        data["t"] = data.index

        data["y"] = data["SDGE"]

        data = data.reset_index(drop=True)

        # convert necessary columns to categorical type
        data["non_working"] = data["non_working"].astype("category")
        data["season"] = data["season"].astype("category")
        data["weekday"] = data["weekday"].astype("category")

        # now convert the categorical columns to codes (0, 1, 2, etc.)
        data["non_working"] = data["non_working"].cat.codes
        data["season"] = data["season"].cat.codes
        data["weekday"] = data["weekday"].cat.codes

        self.data = data

        return self.data


class SalesData(RealLifeTS):
    def __init__(self, root="./datasets/rossman_sales") -> None:
        super().__init__()

        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.num_lags = 5
        self.y_lag_features = [f"y_lag_{i}" for i in range(1, self.num_lags + 1)]
        self.features = ["DayOfWeek", "Customers", "StateHoliday"]

        self.y_lag_diff_features = [f"{feature}_diff" for feature in self.y_lag_features]
        self.diff_features = [f"{feature}_diff" for feature in self.features]

        self.nfeatures = len(self.features) + self.num_lags

        self.period_length = 7
        self.region_width_local = 1
        self.region_width_exp = 3

        self.exponential = 0.4

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a csv file

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        df = pd.read_csv(self.root + "/data.csv", dtype={"StateHoliday": str, "Date": str})

        ###########################################
        # Preprocessing

        # remove store specific columns
        df = df.drop(["Promo", "Open", "SchoolHoliday"], axis=1)

        # change dtype from object to float
        for col in df.columns:
            if df[col].dtype == "object" and col != "Date":
                if col == "StateHoliday":
                    df[col] = df[col].replace({"0": 0, "a": 1, "b": 2, "c": 3})

                df[col] = df[col].astype("int")

        # we group by date and take the mean of everything
        df = (
            df.groupby("Date")
            .agg({"Sales": "mean", "Customers": "mean", "DayOfWeek": "median", "StateHoliday": "max"})
            .reset_index()
        )

        # convert values to int
        df["Sales"] = df["Sales"].astype("int")
        df["Customers"] = df["Customers"].astype("int")
        df["DayOfWeek"] = df["DayOfWeek"].astype("int")
        df["StateHoliday"] = df["StateHoliday"].astype("int")

        # reset index to datetime
        df.index = pd.to_datetime(df["Date"])

        # drop date
        df = df.drop(columns=["Date"])

        # strip all column names from white spaces
        df.columns = df.columns.str.strip()

        # sort the series by date
        df = df.sort_index()

        # set column names correctly
        df["t"] = df.index

        df["y"] = df["Sales"]

        df = df.reset_index(drop=True)

        self.data = df

        return self.data


class AirQuality(RealLifeTS):
    def __init__(self, root="./datasets/bejing_air_quality") -> None:
        super().__init__()

        self.root = root

        self.t = None
        self.y = None
        self.data = None

        self.num_lags = 5
        self.y_lag_features = [f"y_lag_{i}" for i in range(1, self.num_lags + 1)]
        self.features = [
            "year",
            "month",
            "day",
            "hour",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "wd",
            "WSPM",
        ]

        self.distance_features = [
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "WSPM",
        ]  # no categorical features

        self.y_lag_diff_features = [f"{feature}_diff" for feature in self.y_lag_features]
        self.diff_features = [f"{feature}_diff" for feature in self.features]

        self.nfeatures = len(self.features) + self.num_lags

        self.period_length = 24 * 7  # weekly on hourly data -> 168
        
        self.region_width_local = 5 
        self.region_width_exp = 83 
        self.exponential = 0.95
        
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def load(self):
        """Load the data from a csv file

        Specifically,
        also set the t, y, trend, seasonality, and noise attributes
        """
        data = pd.read_csv(self.root + "/PRSA_Data_Tiantan_20130301-20170228.csv")

        # convert wd column to categorical
        data["wd"] = data["wd"].astype("category")
        data["wd"] = data["wd"].cat.codes

        # drop station column
        data = data.drop(["station", "No"], axis=1)

        data.interpolate(inplace=True)

        # use year, month, day, hour as index
        data.index = pd.to_datetime(data[["year", "month", "day", "hour"]])
        data.sort_index(inplace=True)

        data["t"] = data.index
        data["y"] = data["PM2.5"]

        data = data.reset_index(drop=True)

        self.data = data

        return self.data

def get_data(args):
    """returns the time series data and the data splits"""
    
    if args.dataset == "synthetic":
        LENGTH = 15 * 365  # 15 years
        time_series = Synthetic(length=LENGTH)
        time_series.create_time_series(intercept=0, slope=0.1, amplitude=100, noise_std=1, periods=["monthly"])
        time_series.create_lags_y()

        # round to 10 decimal places to prevent floating point errors
        time_series.original = time_series.original.round(10)
        time_series.trend = time_series.trend.round(10)
        time_series.seasonality = time_series.seasonality.round(10)
        time_series.noise = time_series.noise.round(10)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = time_series.num_lags
        N_DIFF = 1

    elif args.dataset == "temperature":
        time_series = TemperatureTS()
        time_series.create_time_series(seasonality="yearly")
        time_series.create_diff_features()
        time_series.create_lags_y()

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = time_series.num_lags
        N_DIFF = 1

    elif args.dataset == "natural-gas":
        time_series = NaturalGasTS()
        time_series.create_time_series(seasonality="monthly")
        time_series.create_lags_y()

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = 1
        N_DIFF = 1

        time_series.plot_pattern(show_plot=False)

    elif args.dataset == "DJIA":
        time_series = DJIATS()
        time_series.create_time_series(seasonality="semi-annually")

        # create lags for the time series
        time_series.create_lags_y()

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = 1
        N_DIFF = 1

        time_series.plot_pattern(show_plot=False)

    elif args.dataset == "energy-consumption":
        time_series = EnergyConsumption()
        time_series.create_time_series(seasonality="half_daily", frequency="hourly")
        time_series.create_diff_features()
        time_series.create_lags_y()
        # create a column with the difference between the current and previous value to add columns of the stationary time series

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        time_series.plot_pattern(show_plot=False)

        N_LAG = time_series.num_lags
        N_DIFF = 1

    elif args.dataset == "sales":
        time_series = SalesData()
        time_series.create_time_series(seasonality="weekly")
        time_series.create_diff_features()
        time_series.create_lags_y()

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = time_series.num_lags
        N_DIFF = 1

    elif args.dataset == "air-quality":
        time_series = AirQuality()
        time_series.create_time_series(seasonality="weekly", frequency="hourly")
        time_series.create_diff_features()
        time_series.create_lags_y()

        # get original data length (unlagged data)
        LENGTH = len(time_series.data)

        # create all data splits
        SIZES = [0.6, 0.2, 0.2]
        data_splits = Data(
            time_series.original, time_series.trend, time_series.seasonality, time_series.noise, sizes=SIZES
        )

        # check whether the length of the data is correct
        N_LAG = time_series.num_lags
        N_DIFF = 1

    else:
        raise ValueError("Invalid dataset")

    assert len(data_splits.original.train) + len(data_splits.original.cal) + len(
        data_splits.original.test
    ) == LENGTH - (N_LAG + N_DIFF)
    assert len(data_splits.trend.train) + len(data_splits.trend.cal) + len(data_splits.trend.test) == LENGTH - (
        N_LAG + N_DIFF
    )
    assert len(data_splits.seasonal.train) + len(data_splits.seasonal.cal) + len(
        data_splits.seasonal.test
    ) == LENGTH - (N_LAG + N_DIFF)
    assert len(data_splits.noise.train) + len(data_splits.noise.cal) + len(data_splits.noise.test) == LENGTH - (
        N_LAG + N_DIFF
    )

    fig, ax = time_series.plot_decomposition(show_plot=False)
    return time_series, data_splits, fig, ax
