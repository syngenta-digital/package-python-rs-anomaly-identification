# General imports
import sys
import os
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)

# class specific imports
from src import credentials, data_downloader
from scripts import user_config as user_config

if __name__ == "__main__":
    os.environ['SH_CLIENT_SECRET'] = credentials.CREDENTIALS['SH_CLIENT_SECRET']
    os.environ['SH_CLIENT_ID'] = credentials.CREDENTIALS['SH_CLIENT_ID']
    os.environ['SH_INSTANCE_ID'] = credentials.CREDENTIALS['SH_INSTANCE_ID']

    filename = user_config.FEATURE_COLLECTION_FILE
    downloader = data_downloader.DataDownloader(filename)
    downloader.download(range_dates=user_config.RANGE_DATES, params=user_config.PARAMS,
                        output_path='../data/raw/nematode_fields/')
    print("Success!")
