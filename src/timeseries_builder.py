# General imports
import numpy as np
from scipy.signal import savgol_filter


class TimeSeriesBuilder:
    @staticmethod
    def compute_sg_time_series_interp(time_series: list[float], doys: list[int], interp_params: dict) -> list[float]:
        """
        Creates an interpolated time series using linear interpolation and Savitzky-Golay filter.

        :param time_series: (list[float]) non interpolated input time series.
        :param doys: (list[int]) input days of the year.
        :param interp_params: (dict) dictionary with interpolation parameters such as window_length and polyorder.
        :return: (list[float]) interpolated output time series.
        """

        nan_indexes = [time_series.index(x) for x in time_series if np.isnan(x)]

        time_series_clean = [time_series[i] for i in range(0, len(time_series)) if i not in nan_indexes]
        doys_clean = [doys[i] for i in range(0, len(doys)) if i not in nan_indexes]
        x_interp = [x for x in range(1, interp_params['days_interp'] + 1)]
        time_series_interp = np.interp(x_interp, doys_clean, time_series_clean)

        sg_time_series_interp = savgol_filter(time_series_interp, window_length=interp_params['window_length'],
                                              polyorder=interp_params['polyorder'])
        return list(sg_time_series_interp)


