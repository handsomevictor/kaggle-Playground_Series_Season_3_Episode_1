import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from data_processing import Data


def find_city_name(lati, long):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(f"{lati}, {long}")
    # print(location.raw['address'])

    try:
        neighborhood = location.raw['address']['neighbourhood']
    except:
        neighborhood = None
    try:
        suburb = location.raw['address']['suburb']
    except:
        suburb = None
    try:
        county = location.raw['address']['county']
    except:
        county = None
    try:
        city = location.raw['address']['city']
    except:
        city = None
    return neighborhood, suburb, county, city


# def get_geo_info_concurrent(max_workers=10):
#     # 不允许这样，超过最大request limit
#     train_df = pd.read_csv('train.csv', index_col=0)
#     with ProcessPoolExecutor(max_workers=50) as pool:
#         res = list(tqdm(pool.map(find_city_name, train_df['Latitude'], train_df['Longitude']), total=len(train_df)))
#     return res


if __name__ == '__main__':
    train_df = Data('train', scaler_type='none', use_original_data=True).data
    train_df = train_df.reset_index()
    # add neighborhood and county name
    for i in tqdm(range(len(train_df))):
        lati = train_df.iloc[i]['Latitude']
        long = train_df.iloc[i]['Longitude']
        neighborhood, suburb, county, city = find_city_name(lati, long)
        train_df.loc[i, 'Neighborhood'] = neighborhood
        train_df.loc[i, 'County'] = county
        train_df.loc[i, 'Suburb'] = suburb
        train_df.loc[i, 'City'] = city

    train_df.to_csv('train_add_geo_info.csv')

    prediction_data = Data('test', scaler_type='none').processed_data
    prediction_data = prediction_data.reset_index()

    for i in tqdm(range(len(prediction_data))):
        lati = prediction_data.iloc[i]['Latitude']
        long = prediction_data.iloc[i]['Longitude']
        neighborhood, suburb, county, city = find_city_name(lati, long)

        prediction_data.loc[i, 'Neighborhood'] = neighborhood
        prediction_data.loc[i, 'County'] = county
        prediction_data.loc[i, 'Suburb'] = suburb
        prediction_data.loc[i, 'City'] = city

    prediction_data.to_csv('test_add_geo_info.csv')
