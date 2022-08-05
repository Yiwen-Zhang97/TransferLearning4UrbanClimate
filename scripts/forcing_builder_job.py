import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import os
import ee
from osgeo import gdal
import cesm_utils
import datetime
import pickle
import sys
sys.path.append('/glade/u/home/yiwenz/.local/lib/python3.9/site-packages')

#Change the following:
city_idx_start,city_idx_end=[45,60]


dataDir='/glade/scratch/yiwenz/CESM_0.125_US_raw/'
outDir_forcing='/glade/scratch/yiwenz/CESM_0.125_US_cities'
outDir_modis_grid='/glade/scratch/yiwenz/modis_lat_lon_US_cities'
us_city_collection = 'users/yiwenz9/us_cities'
shp_file = "tl_2020_us_uac10/tl_2020_us_uac10.shp"
city_400_file='/glade/u/home/yiwenz/TransferLearning/Forcing_scripts/largest_400_cities.csv'
var=["FLDS", "FSDS", "PRECTmms", "PSRF", "QBOT", "TBOT", "WIND"]
time_len = 6024

ee.Initialize()
modis=ee.ImageCollection('MODIS/061/MYD11A1').filterDate('2002-09-01')
cities = ee.FeatureCollection(us_city_collection)

#Specify timezone
EST,CST,MST,PST=[18,19,20,21]
#city_name,city_num,city_hour
cities_400=pd.read_csv(city_400_file)[['City_Name','NAME10','Time_zone']]

for city_idx in range(city_idx_start,city_idx_end):
    city_list=cities_400.loc[city_idx].to_list()
    city_name, city_num, city_hour=city_list
    city_hour = vars()[city_hour]
    print(f'Processing: {city_name}, name in cities shapefile:{city_num}, coversion to utc:{city_hour}.')
    PATH_TO_STORE_PICKLE=os.path.join(outDir_forcing,'forcing_'+city_name+'.pkl')
    PATH_TO_STORE_MODIS_LAT_LON=os.path.join(outDir_modis_grid,'modis_grid_'+city_name+'.csv')
    target_city=cities.filter(ee.Filter.eq('NAME10',city_num))
    df_modis = cesm_utils.get_modis_vector(modis,target_city)
    if not os.path.exists(PATH_TO_STORE_MODIS_LAT_LON):
        df_modis.to_csv(PATH_TO_STORE_MODIS_LAT_LON)
    if not os.path.exists(PATH_TO_STORE_PICKLE):
        gdf_modis = gpd.GeoDataFrame(df_modis, geometry=gpd.points_from_xy(df_modis.lon, df_modis.lat))
        #FSDS and TPQWL have different forms of timestamp so the method of selecting the hour is a little different.
        #PRECT is already daily sum data, so don't need to choose hours.
        xr_fsds = xr.open_dataset(os.path.join(dataDir,'Solar.daily.nc'))
        xr_fsds = xr_fsds.sel(time=xr_fsds.time.dt.round('H').dt.hour == city_hour)
        xr_TPQWL = xr.open_dataset(os.path.join(dataDir,'TPQWL.daily.nc'))
        xr_TPQWL = xr_TPQWL.sel(time=xr_TPQWL.time.dt.hour == city_hour)
        xr_PRECT = xr.open_dataset(os.path.join(dataDir,'Precip.daily.nc'))
        pickle_data={}
        for time_idx in range(time_len):    
            df_allvar,date = cesm_utils.get_daily_var(dataDir, xr_fsds, xr_TPQWL,xr_PRECT, time_index=time_idx)
#            print(date)
            gdf_allvar = gpd.GeoDataFrame(df_allvar, geometry=gpd.points_from_xy(df_allvar.LONGXY, df_allvar.LATIXY))
            gdf_join = gdf_modis.sjoin_nearest(gdf_allvar, how='left', distance_col="distances")[var]
            if date not in pickle_data.keys():
                pickle_data[date] = {}
                for idx in gdf_join.index:
                    forcing_list = np.array(gdf_join.loc[idx][var].tolist(),dtype=np.float32)
                    pickle_data[date][idx] = forcing_list                    
        with open(PATH_TO_STORE_PICKLE, 'wb') as f:
            pickle.dump(pickle_data, f)
        print(f'Saved to {PATH_TO_STORE_PICKLE}.')
    else:
        print(f'Forcing file already exists for {city_name}')
