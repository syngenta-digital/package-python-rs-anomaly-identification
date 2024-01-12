import glob
import numpy as np
import xarray as xr
import datetime


def load_data(path: str) -> dict[xr.core.dataset.Dataset]:
    files = glob.glob(path + "*.nc")
    files = np.sort(files)

    field_data_dict = {}
    for file in files:
        key = file.split("/")
        key = key[len(key) - 1]
        # field_data = xr.open_dataset(file)
        field_data_dict[key] = xr.open_dataset(file)
    return field_data_dict


def compute_dates(field_data: xr.core.dataset.Dataset) -> list[datetime]:
    dates = []
    for t in range(0, len(field_data.time)):
        date_format = '%Y-%m-%d'
        date_obj = field_data['time'][t].time.dt.date.item()
        dates.append(date_obj.strftime(date_format))
    return dates


def create_ndvi(ds: xr.core.dataset.Dataset, time: int) -> np.array:
    ndvi = (ds['B08'][time].data - ds['B04'][time].data)/(ds['B08'][time].data + ds['B04'][time].data)
    return ndvi


def create_rasters(field_data: xr.core.dataset.Dataset) -> list[np.ndarray]:
    rasters = []
    for t in range(0, len(field_data.time)):
        raster = create_ndvi(field_data, t)
        rasters.append(raster)
    return rasters
