import os
import glob
import geojson
import shapely
from shapely.geometry import shape
from cads.tools import SenHub, tools, params, VegIndices
import config as config
from credentials import CREDENTIALS


class DataDownloader:
    def __init__(self, filename: str) -> None:
        """
        Contructor of the class: it loads the feature collection file indicated in "filename".
        :param filename: (str) name of the feature collection file.

        """
        with open(filename) as f:
            self.feature_collection = geojson.load(f)

    def get_feature_collection(self) -> None:
        """
        Returns feature collection loaded.
        :return:

        """
        return self.feature_collection

    def get_polygon(self, index: int) -> shapely.geometry.multipolygon.MultiPolygon:
        """
        Returns geometry of the i-th polygon in the feature collection.
        :param index: (int) Index of the i-th polygon in the feature collection.
        :return: (MultiPolygon) Shapely geometry.
        """

        poly = {'type': 'MultiPolygon',
                'coordinates': [self.feature_collection['features'][index]['geometry']['coordinates']]}
        return shape(poly)

    def get_downloaded_fields_list(self, path: str) -> list[str]:
        """
        Returns a list of indexes of files already downloaded in the folder indicated by "path".
        :param path: (str) Path of the folder containing downloaded files.
        :return: (list[int]) list of indexes of files already downloaded.

        """
        already_downloaded_fields = glob.glob(f"{path}*.nc")
        already_downloaded_fields = [x.split('field_')[1] for x in already_downloaded_fields]
        already_downloaded_fields = [x.split('.')[0] for x in already_downloaded_fields]
        return list(set(already_downloaded_fields))

    def download(self, range_dates: list[str], params: dict, output_path: str, force=False) -> None:
        """
        Download data in the specified range of dates
        :param range_dates: (list[str]) List containing start and end dates for downloading
        :param params: (dict) Dictionary with parameters used for download (cloud percentage, satellite name, etc.).
        :param output_path: (str) Output folder where downloaded files will be saved to.
        :param force: (Bool) Flag to force or prevent redownload.
        :return:

        """
        season_start = int(range_dates[0].split('-')[0])
        season_end = int(range_dates[1].split('-')[0])
        already_downloaded_fields = self.get_downloaded_fields_list(output_path)
        for feature_index in range(0, len(self.feature_collection['features'])):
            polygon = self.get_polygon(index=feature_index)
            if not force:
                if str(feature_index) in already_downloaded_fields:
                    print(f"Field {feature_index} already downloaded.")
                    continue
            for season in range(season_start, season_end):
                satimgs = SenHub(sat=params['SAT'], product='', sitename='field_' + str(feature_index),
                                 outdir=output_path, aoi=polygon,
                                 start_date=str(season) + '-01-01', end_date=str(season) + '-12-31',
                                 cloud_frac_tile_max=params['CLOUD_FRAC_TILE_MAX'],
                                 cloud_frac_field_max=params['CLOUD_FRAC_FIELD_MAX'],
                                 clp_threshold=params['CLP_THRESHOLD'])
                satimgs.request()


if __name__ == "__main__":
    os.environ['SH_CLIENT_SECRET'] = CREDENTIALS['SH_CLIENT_SECRET']
    os.environ['SH_CLIENT_ID'] = CREDENTIALS['SH_CLIENT_ID']
    os.environ['SH_INSTANCE_ID'] = CREDENTIALS['SH_INSTANCE_ID']

    filename = config.FEATURE_COLLECTION_FILE
    downloader = DataDownloader(filename)
    downloader.download(range_dates=config.RANGE_DATES, params=config.PARAMS,
                        output_path='../data/raw/nematode_fields/')
    print("Success!!")
