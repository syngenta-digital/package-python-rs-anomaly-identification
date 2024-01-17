# General imports
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Class specific imports
from src.config import DAYS_INTERP, MODELS_PARAMS
from src.timeseries_builder import TimeSeriesBuilder
import planting_date as planting_date


class NormalizationBuilder:
    def __init__(self, models_config: dict = None):
        """
        Constructor of the class: it loads the pretrained models used by planting_date class.

        :param models_config: dictionary with initalization parameters.
        """

        if models_config is None:
            models_config = MODELS_PARAMS
        self.test_1reg_cp_alpha = models_config['TEST_1REG_CP_ALPHA']
        self.fitted_pre_clf = joblib.load(f"{models_config['MODELS_PATH']}/{models_config['TEST_PRECLF_FILE']}")
        self.fitted_low_reg = joblib.load(f"{models_config['MODELS_PATH']}/{models_config['TEST_3REG_FILES_LOW']}")
        self.fitted_mid_reg = joblib.load(f"{models_config['MODELS_PATH']}/{models_config['TEST_3REG_FILES_MID']}")
        self.fitted_high_reg = joblib.load(f"{models_config['MODELS_PATH']}/{models_config['TEST_3REG_FILES_HIGH']}")

    def create_planting_date_dataframe(self, ndvi: list[float], dates: list[str], lon_lat: tuple[float, float]) \
            -> pd.DataFrame:
        """
        Creates the dataframe used as input to the planting_date package.

        :param ndvi: list[float] time series of all seasons chronologically sorted.
        :param dates: list[str] list of all correspondent dates of the time series values, also chronologically sorted.
        The only accepted format is "YYY-mm-dd".
        :param lon_lat: (tuple[float, float]) a tuple of (lon, lat) belonging to the field under analysis.
        :return: (pd.DataFrame) a pandas dataframe to be used with planting_date package.
        """

        try:
            doys = [datetime.strptime(dates[i], "%Y-%m-%d").timetuple().tm_yday for i in range(len(dates))]
        except ValueError:
            print("Incorrect data format, should be YYYY-MM-DD.")
            exit()

        season_windows = self.create_season_windows(dates)

        cropzones_interp, dates_interp, ndvi_interp, lon_interp, lat_interp, doys_interp = [], [], [], [], [], []
        seasons = list(season_windows.keys())
        for season in seasons:
            start = season_windows[season][0]
            end = season_windows[season][1]

            dates_interp += [self.get_date_from_doy(doy=i, year=int(season)) for i in range(1, DAYS_INTERP + 1)]
            ndvi_interp += TimeSeriesBuilder.compute_sg_time_series_interp(time_series=ndvi[start:end],
                                                                           doys=doys[start:end],
                                                                           interp_params={'window_length': 50,
                                                                                          'polyorder': 3,
                                                                                          'days_interp': 365})
            cropzones_interp += [season] * DAYS_INTERP
            lon_interp += [lon_lat[0]] * DAYS_INTERP
            lat_interp += [lon_lat[1]] * DAYS_INTERP
            doys_interp += [i for i in range(1, DAYS_INTERP + 1)]

        df = pd.DataFrame()
        df = df.assign(date=dates_interp, NDVI=ndvi_interp, cropzone=cropzones_interp,
                       longitude=lon_interp, latitude=lat_interp, DOY=doys_interp)
        return df

    def normalize_time_series(self, ndvi: list[float], dates: list[str], lon_lat: tuple[float, float]) -> dict:
        """
        Interface with the planting_date package, returning planting dates for each season, as well as computed deltas
        used to align them.

        :param ndvi: (list[float]) time series of all seasons chronologically sorted.
        :param dates: (list[str]) list of all correspondent dates of the time series values, also chronologically
        sorted.
        :param lon_lat: (tuple[float, float]) a tuple of (lon, lat) belonging to the field under analysis.
        :return: ([dict]) dict composed of two keys: planting dates and deltas. Planting dates and deltas are computed
        for each season.
        """

        planting_date_input_df = self.create_planting_date_dataframe(ndvi=ndvi, dates=dates,
                                                                     lon_lat=(lon_lat[0], lon_lat[1]))

        # Estimate planting_dates
        ndvi_prepd_df, ndvi_preds_df = self.estimate_planting_dates(planting_date_input_df=planting_date_input_df)

        # Create planting_date dict
        planting_dates_dict = self.create_planting_date_dict(ndvi_prepd_df=ndvi_prepd_df, ndvi_preds_df=ndvi_preds_df)

        # Compute Deltas
        shift_dict = self.compute_deltas(planting_dates_dict=planting_dates_dict)

        return shift_dict

    def estimate_planting_dates(self, planting_date_input_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Estimates the planting dates using the planting_date package.

        :param planting_date_input_df: (pd.Dataframe) input dataframe.
        :return: (pd.DataFrame, pd.DataFrame): dataframes obtained after preparation and prediction steps.
        """

        # Prepare data using planting date package
        prepper = planting_date.LandDBPrep()
        ndvi_prepd_df = prepper.prep(planting_date_input_df, for_training=False, progress=True)

        # Predict planting date
        predictor = planting_date.PDModelPred()
        ndvi_preds_df = predictor.predict_3reg(ndvi_prepd_df, alpha=self.test_1reg_cp_alpha,
                                               fitted_pre_clf=self.fitted_pre_clf,
                                               fitted_low_reg=self.fitted_low_reg,
                                               fitted_mid_reg=self.fitted_mid_reg,
                                               fitted_high_reg=self.fitted_high_reg)

        return ndvi_prepd_df, ndvi_preds_df

    @staticmethod
    def compute_deltas(planting_dates_dict: dict) -> dict:
        """
        Creates a dictionary with planting dates and deltas.

        :param planting_dates_dict: dictionary with key="season" and value="predicted planting date".
        :return: (dict) dict composed of two keys: planting dates and deltas. Planting dates and deltas are computed
        for each season.
        """

        sorted_pd_dates = dict(sorted(planting_dates_dict.items(), key=lambda item: item[1]))
        first_year = list(sorted_pd_dates.keys())[0]
        first_pd = sorted_pd_dates[first_year]

        deltas_dict = {}
        for season in planting_dates_dict.keys():
            if planting_dates_dict[season] is not None:
                delta = sorted_pd_dates[season] - first_pd
                deltas_dict[season] = delta

        shift_dict = {'planting_dates': planting_dates_dict, 'deltas': deltas_dict}

        return shift_dict

    @staticmethod
    def create_planting_date_dict(ndvi_prepd_df: pd.DataFrame, ndvi_preds_df: pd.DataFrame) -> dict:
        """
        Creates a dictionary with key="season" and value="predicted planting date".

        :param ndvi_prepd_df: (pd.DataFrame) dataframe created after preparation step.
        :param ndvi_preds_df: (pd.DataFrame) dataframe created after prediction step.
        :return: (dict) dictionary with key="season" and value="predicted planting date".
        """

        dates = list(ndvi_prepd_df['cropzone'])
        years_prep = [x for x in dates]
        pd_dates = {}
        preds_values = list(ndvi_preds_df.sort_index()['y_hat'])
        for i in range(len(years_prep)):
            pd_dates[years_prep[i]] = preds_values[i]

        return pd_dates

    @staticmethod
    def create_season_windows(dates: list[str]) -> dict[[int, int]]:
        """
        Based on the sorted dates array, finds the season windows composed of [min, max] indexes.

        :param dates: (list[str]) sorted dates.
        :return: (dict[int, int]) dictionary composed key="season" and value="correspondent season window indexes".
        """

        years = list(np.sort([x.split('-')[0] for x in dates]))
        seasons = list(np.sort(np.unique(years)))
        season_windows = {}
        for i in range(0, len(seasons) - 1):
            current_season = seasons[i]
            next_season = seasons[i + 1]
            season_windows[seasons[i]] = [years.index(current_season),
                                          years.index(next_season)]
        last_year = seasons[len(seasons) - 1]
        season_windows[last_year] = [years.index(last_year), len(years)]

        return season_windows

    @staticmethod
    def get_date_from_doy(doy: int, year: int) -> datetime:
        """
        Return the correspondent datetime for a given doy and year.

        :param doy: (int) day of the year.
        :param year: (int) year.
        :return: (datetime) correspondent datetime.
        """

        return datetime(year, 1, 1) + timedelta(doy - 1)
