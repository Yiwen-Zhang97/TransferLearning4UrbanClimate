#Define map plotting function
#import cartopy.crs as ccrs
#from cartopy.util import add_cyclic_point
from matplotlib import cm
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import cartopy.feature as cfeature
import xarray as xr
import os
import pandas as pd
import geopandas as gpd
import shapely
import datetime

def get_daily_var(xr_TPQWLS,xr_PRECT,time_index):
    
    xr_TPQWLS = xr_TPQWLS.isel(time=time_index)
    df_TPQWLS = xr_TPQWLS.to_dataframe().reset_index().drop(['x','y','time'],axis=1)
    
    xr_PRECT = xr_PRECT.isel(time=time_index)
    df_PRECT = xr_PRECT.to_dataframe().reset_index().drop(['x','y','time'],axis=1)

    df_allvar = df_TPQWLS.merge(df_PRECT,on=['lat','lon'])
    
    time_TPQWL=str(xr_TPQWLS.time.dt.date.item())[:10]
    time_PRECT=str(xr_PRECT.time.dt.date.item())[:10]
    if time_TPQWL==time_PRECT:
        date=time_TPQWL
    else:
        print('Time does not match.')
        print(time_TPQWL,time_PRECT)
    
    return df_allvar,date

def reproject_collection(image):
    return image.reproject('EPSG:32612',scale=1000)

def clip_collection(image,target_city):
    return image.clip(target_city)

def get_modis_vector(img_col,target_city):
    modis_reproj=img_col\
            .map(lambda image: clip_collection(image, target_city))\
            .filterBounds(target_city)\
            .select('LST_Day_1km','QC_Day')
    vector=modis_reproj.first().sample(
      region= target_city.geometry(),
      geometries=True,
      dropNulls=False,
      scale=1000
    )
    vector_data = vector.geometry().coordinates().getInfo()
    df_modis = pd.DataFrame(vector_data).rename(columns={0: "lon", 1: "lat"})
    return df_modis

def download_nc(fname,df_modis,df_allvar,var,ftime):
    gdf_modis = gpd.GeoDataFrame(df_modis, geometry=gpd.points_from_xy(df_modis.lon, df_modis.lat))
    gdf_allvar = gpd.GeoDataFrame(df_allvar, geometry=gpd.points_from_xy(df_allvar.LONGXY, df_allvar.LATIXY))
    gdf_join = gdf_modis.sjoin_nearest(gdf_allvar, how='left', distance_col="distances")[var]
#    time_arr=datetime.datetime.strptime(time.format("YYYY-MM-dd").getInfo(), '%Y-%m-%d')
    f_join=gdf_join.to_xarray()
    f_join=f_join.expand_dims({'time':[ftime]})
    f_join.to_netcdf(fname)
    return f_join

def proc_data(ds,var):
    ds['time'] = ds.time.dt.round('H')
    ds = ds.sel(time=ds.time.dt.hour.isin([20,21]))
    ds[var] = ds[var].groupby('time.day').mean()
    ds = ds[['LONGXY','LATIXY',var]]
    ds['LONGXY'] = (ds.LONGXY + 180) % 360 - 180
    return ds

def get_stats(ds):
    """
    ds: xarray
    """
    total_num = ds.size
    zero_num = total_num - np.count_nonzero(ds)
    nan_num = np.count_nonzero(np.isnan(ds))
    value_num = total_num - zero_num - nan_num
    zero_per = round(zero_num/total_num*100,2)
    nan_per = round(nan_num/total_num*100,2)
    value_per = round(value_num/total_num*100,2)
    print(f'There are {zero_num} zero values, acccounting for {zero_per}% of the dataset.')
    print(f'There are {nan_num} nan values, acccounting for {nan_per}% of the dataset.')
    print(f'There are {value_num} with other values, acccounting for {value_per}% of the dataset.')
    xr.plot.hist(ds.where(ds!=0), bins=100)
    plt.title('Dataset with non-zero values')
    plt.show()

def plot_map(data, lon, lat, title='Emulated Temperature',cmaps='viridis', label='K', level = np.arange(0.0,5.1,0.01), cb_ticks=np.arange(0,5.5,0.5)):
    top = plt.cm.get_cmap('Oranges_r', 128)
    bottom = plt.cm.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', newcolors)
    if cmaps == 'viridis':
        cmap=plt.cm.get_cmap('viridis', 256)
    elif cmaps == 'ratio':
        cmap=my_cmap

#    data_c, lon_c = add_cyclic_point(data, coord=lon)
    fig=plt.figure(figsize=(16,9))
    ax = plt.axes(projection=ccrs.PlateCarree())
#    ax.set_global()
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='110m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')
    
    ax.coastlines(resolution='110m',color='silver')
    gl=ax.gridlines(linestyle='--', draw_labels=True)
    gl.top_labels  = False
    gl.right_labels  = False
#    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    maps=plt.contourf(lon, lat, data,levels=level, transform=ccrs.PlateCarree(),extend='both', cmap=cmap)
    plt.title(title,fontsize=25)
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.04])
    cbar=plt.colorbar(maps, cax=cbar_ax, orientation="horizontal",ticks=cb_ticks)
    cbar.set_label(label=label, fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    fig.text(0.5, 0.08, 'Longitude(°)', ha='center', fontsize=25)
    fig.text(0.05, 0.5, 'Latitude(°)', va='center', rotation='vertical', fontsize=25)


def regrid_data(cropped_df,method='bilinear'):
    ds_out = xe.util.grid_2d((cropped_df.lon.min()+0.01), (cropped_df.lon.max()-0.01), 0.01, (cropped_df.lat.min()+0.01), (cropped_df.lat.max()-0.01), 0.01)
    regridder = xe.Regridder(cropped_df, ds_out, method, periodic=True, reuse_weights=False)
    ds_merge_ts_reg = regridder(cropped_df,keep_attrs=True)
    print(f"******Regridded forcing******")
    return ds_merge_ts_reg

