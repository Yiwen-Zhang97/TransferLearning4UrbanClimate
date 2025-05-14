import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import sys
from tqdm import tqdm
import xesmf as xe
from cesm_utils_ERA5 import get_daily_var
import pickle
sys.path.append('/glade/u/home/yiwenz/.local/lib/python3.9/site-packages')

#Change the following:
# city_idx_start,city_idx_end=[322,383]
city_idx_all=[47,73,76,129,133]

dataDir='/glade/work/yiwenz/ERA5_0.25_US_All/'
outDir_forcing_day='/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Day_Regrid_US_All'
outDir_forcing_night='/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Night_Regrid_US_All'
outDir_modis_grid='/glade/work/yiwenz/modis_lat_lon_US_926'
city_400_file='/glade/u/home/yiwenz/TransferLearning/City_NameList/US_384.csv'
variables=["ssrd", "strd", "tp", "sp", "rh", "temperature", "uwind","vwind"]

#Specify timezone
#(Day,Night)
day_hour=13
night_hour=1
xr_TPQWLS=xr.open_dataset(os.path.join(dataDir,f'TPQWLS.nc'))
xr_tp=xr.open_dataset(os.path.join(dataDir,f'tp.nc'))
cities_400=pd.read_csv(city_400_file)

def crop_df(df,df_modis):
    min_lat=df_modis.lat.min()
    max_lat=df_modis.lat.max()
    min_lon=df_modis.lon.min()
    max_lon=df_modis.lon.max()
    mask_lon = (df.lon >= (min_lon-0.3)) & (df.lon <= (max_lon+0.3))
    mask_lat = (df.lat >= (min_lat-0.3)) & (df.lat <= (max_lat+0.3))
    df_cropped = df.where(mask_lon & mask_lat, drop=True)
    return df_cropped

def regrid_data(cropped_df,city_name,method='patch'):
    ds_out = xe.util.grid_2d((cropped_df.lon.min()+0.01), (cropped_df.lon.max()-0.01), 0.01, (cropped_df.lat.min()+0.01), (cropped_df.lat.max()-0.01), 0.01)
    regridder = xe.Regridder(cropped_df, ds_out, method, periodic=True, reuse_weights=False,filename=f'/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Forcing/{method}_{city_name}.nc')
    ds_merge_ts_reg = regridder(cropped_df,keep_attrs=True)
    print(f"******Regridded forcing******")
    return ds_merge_ts_reg

for city_idx in city_idx_all:#range(city_idx_start,city_idx_end):
    try:
        city_name=str(int(cities_400.loc[city_idx].City_Name))
    except:
        city_name=cities_400.loc[city_idx].City_Name
    PATH_TO_STORE_MODIS_LAT_LON=os.path.join(outDir_modis_grid,'modis_grid_'+city_name+'.csv')
    df_modis = pd.read_csv(PATH_TO_STORE_MODIS_LAT_LON)
    time_offset=round(((df_modis.lon.max()+df_modis.lon.min())/2)/15)
    city_hour_day = day_hour-time_offset
    city_hour_night = night_hour-time_offset
    print('--------------------------------------------------------------------------------------------')
    print(f'Processing: {city_name}, coversion to utc: day {city_hour_day}, night {city_hour_night}.')
    PATH_TO_STORE_PICKLE_DAY=os.path.join(outDir_forcing_day,'forcing_'+city_name+'.pkl')
    PATH_TO_STORE_PICKLE_NIGHT=os.path.join(outDir_forcing_night,'forcing_'+city_name+'.pkl')
    if not os.path.exists(PATH_TO_STORE_PICKLE_DAY) and not os.path.exists(PATH_TO_STORE_PICKLE_NIGHT):
        gdf_modis = gpd.GeoDataFrame(df_modis, geometry=gpd.points_from_xy(df_modis.lon, df_modis.lat))
        if city_hour_day>24:
            xr_TPQWLS_day = xr_TPQWLS.sel(time=xr_TPQWLS.time.dt.hour == city_hour_day-24)
        else:
            xr_TPQWLS_day = xr_TPQWLS.sel(time=xr_TPQWLS.time.dt.hour == city_hour_day)
            xr_TPQWLS_day = regrid_data(crop_df(xr_TPQWLS_day,df_modis),city_name)
        
        if city_hour_night<0:
            xr_TPQWLS_night = xr_TPQWLS.sel(time=xr_TPQWLS.time.dt.hour == city_hour_night+24)
        else:
            xr_TPQWLS_night = xr_TPQWLS.sel(time=xr_TPQWLS.time.dt.hour == city_hour_night)
            xr_TPQWLS_night = regrid_data(crop_df(xr_TPQWLS_night,df_modis),city_name)

        time_delta = np.timedelta64(time_offset, 'h')
        xr_tp_day_night = xr_tp.copy()
        xr_tp_day_night['time'] = xr_tp_day_night['time'] + time_delta
        xr_tp_day_night = xr_tp_day_night.resample(time='1D').sum(dim='time')
        xr_tp_total = regrid_data(crop_df(xr_tp_day_night,df_modis),city_name).isel(time=slice(1,None))
        print(len(xr_tp_total.time))
    
        pickle_data_day={}
        pickle_data_night={}
        for time_idx in range(1,len(xr_tp_total.time)-1):  
            date = str(xr_tp_total.isel(time=time_idx).time.dt.date.item())[:10]
            if city_hour_day>24:
                df_allvar_day,date_day = get_daily_var(xr_TPQWLS_day,xr_tp_total, time_index=time_idx+1)
            else:
                df_allvar_day,date_day = get_daily_var(xr_TPQWLS_day,xr_tp_total, time_index=time_idx)
            
            df_allvar_day = df_allvar_day.dropna()
            gdf_allvar_day = gpd.GeoDataFrame(df_allvar_day, geometry=gpd.points_from_xy(df_allvar_day.lon, df_allvar_day.lat))
            gdf_join_day = gdf_modis.sjoin_nearest(gdf_allvar_day, how='left', distance_col="distances")[variables]
            if date not in pickle_data_day.keys():
                pickle_data_day[date] = {}
                for idx in gdf_join_day.index:
                    forcing_list_day = np.array(gdf_join_day.loc[idx][variables].tolist(),dtype=np.float32)
                    pickle_data_day[date][idx] = forcing_list_day
                    
            if city_hour_night<0:        
                df_allvar_night,date_night = get_daily_var(xr_TPQWLS_night,xr_tp_total, time_index=time_idx-1)
            else:
                df_allvar_night,date_night = get_daily_var(xr_TPQWLS_night,xr_tp_total, time_index=time_idx)
            df_allvar_night = df_allvar_night.dropna()
            gdf_allvar_night = gpd.GeoDataFrame(df_allvar_night, geometry=gpd.points_from_xy(df_allvar_night.lon, df_allvar_night.lat))
            gdf_join_night = gdf_modis.sjoin_nearest(gdf_allvar_night, how='left', distance_col="distances")[variables]
            if date not in pickle_data_night.keys():
                pickle_data_night[date] = {}
                for idx in gdf_join_night.index:
                    forcing_list_night = np.array(gdf_join_night.loc[idx][variables].tolist(),dtype=np.float32)
                    pickle_data_night[date][idx] = forcing_list_night
            print(date,date_day,date_night)

        with open(PATH_TO_STORE_PICKLE_DAY, 'wb') as f:
            pickle.dump(pickle_data_day, f)
        with open(PATH_TO_STORE_PICKLE_NIGHT, 'wb') as f:
            pickle.dump(pickle_data_night, f)
        print(f'Saved to {PATH_TO_STORE_PICKLE_DAY},{PATH_TO_STORE_PICKLE_NIGHT}.')
    else:
        print(f'Forcing file already exists for {city_name}')    
