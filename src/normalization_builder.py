import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import DAYS_INTERP
from src.timeseries_builder import TimeSeriesBuilder


class NormalizationBuilder:

    def create_planting_date_dataframe(self, ndvi: list[float], dates: list[str], lon_lat: tuple[float, float]) \
            -> pd.DataFrame:
        """

        :param ndvi:
        :param dates:
        :param lon_lat:
        :return:
        """

        doys = [datetime.strptime(dates[i], "%Y-%m-%d").timetuple().tm_yday
                for i in range(len(dates))]
        season_windows = self.create_season_windows(dates)

        cropzones_interp = []
        dates_interp = []
        ndvi_interp = []
        lon_interp = []
        lat_interp = []
        doys_interp = []
        seasons = list(season_windows.keys())
        interp_params = {'window_length': 50, 'polyorder': 3}
        for season in seasons:
            start = season_windows[season][0]
            end = season_windows[season][1]

            dates_interp += [self.get_date_from_doy(i, int(season)) for i in range(1, 366)]
            ndvi_interp += TimeSeriesBuilder.compute_sg_time_series_interp(ndvi[start:end], doys[start:end],
                                                                           interp_params)
            cropzones_interp += [season] * DAYS_INTERP
            lon_interp += [lon_lat[0]] * DAYS_INTERP
            lat_interp += [lon_lat[1]] * DAYS_INTERP
            doys_interp += [i for i in range(1, 366)]

        df = pd.DataFrame()
        df['date'] = dates_interp
        df['NDVI'] = ndvi_interp
        df['cropzone'] = cropzones_interp
        df['longitude'] = lon_interp
        df['latitude'] = lat_interp
        df['DOY'] = doys_interp

        return df

    def perform_normalization(self, ndvi: list[float], dates: list[str], lon_lat: tuple[float, float]) -> dict:
        """

        :param ndvi:
        :param dates:
        :param lon_lat:
        :return:
        """
        planting_date_input_df = self.create_planting_date_dataframe(ndvi, dates, (lon_lat[0], lon_lat[1]))
        return planting_date_input_df

    @staticmethod
    def create_season_windows(dates) -> dict[[int, int]]:
        """

        :param dates:
        :return:
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
    def get_date_from_doy(doy: int, year: int):
        return datetime(year, 1, 1) + timedelta(doy - 1)
