import numpy as np
from tfrecords2numpy_landsat_no_idx import TFRecordsParser as TFRecordsParser_Landsat
from tfrecords2numpy_landsat_no_idx import TFRecordsElevation
from tfrecords2numpy_modis_day_night import TFRecordsParser as TFRecordsParser_Modis
import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import webdataset as wds
import xarray as xr
import h5py
import random
from collections import Counter

#1,2,3,4,6,7,10,11,12,13,14,15,46,47,73,76,129,133
#city_idx_start,city_idx_end=[13,16]
city_idx_all=[1]
#[1,2,3,4,6,7,10,11,12,13,14,15,46,47,73,76,129,133]

threshold_cloud=1
THRESH=1
city_400_file='/glade/u/home/yiwenz/TransferLearning/Forcing_scripts/largest_400_cities_new.csv'
cities_400=pd.read_csv(city_400_file)[['City_Name','NAME10','Time_zone']]

#for city_idx in range(city_idx_start,city_idx_end):
for city_idx in city_idx_all:
    city_list=cities_400.loc[city_idx].to_list()
    city_name, _, _=city_list
    print(city_name)
    PATH_TO_DATA_1 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite'
    PATH_TO_DATA_2 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US_2019_2024/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite_2019_2024'
    PATH_TO_FORCING_DAY = "/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Day_Regrid_US_All/forcing_"+city_name+".pkl"
    PATH_TO_FORCING_NIGHT = "/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Night_Regrid_US_All/forcing_"+city_name+".pkl" 
    PATH_TO_STATION_DAY = f"/glade/work/yiwenz/station_data_day_2013-2024/{city_name}/full_{city_name}_station_data.csv"
    PATH_TO_STATION_NIGHT = f"/glade/work/yiwenz/station_data_night_2013-2024/{city_name}/full_{city_name}_station_data.csv"
    PATH_TO_STORE_FINETUNE_DAY = f"/glade/derecho/scratch/yiwenz/finetune_AllCountries/finetune_city_hdf5_day_1_US_qcMoreLess_0-1_noOutliers_2013-2024/{city_name}_{threshold_cloud}_landsat8_finetune_day.h5"
    PATH_TO_STORE_FINETUNE_NIGHT = f"/glade/derecho/scratch/yiwenz/finetune_AllCountries/finetune_city_hdf5_night_1_US_qcMoreLess_0-1_noOutliers_2013-2024/{city_name}_{threshold_cloud}_landsat8_finetune_night.h5" 
    
    PATH_TO_LATLONG = "/glade/work/yiwenz/modis_lat_lon_US_926/modis_grid_"+city_name+".csv"
    PATH_TO_BUILDING_HEIGHT = "/glade/work/yiwenz/building_height_US/building_height_"+city_name+".csv"
    PATH_TO_ELEVATION = "/glade/work/yiwenz/elevation_US/"+city_name+".tfrecord"
    channel_choices = ['Red', 'Green', 'Blue', "NIR", "SWIR1"]

    def extract_data(finetune_day_path,finetune_night_path):
        """
        Extract all data from all TFRecords files and stores as pickled tuples in the format (image, label)
        :return:
        Each Sample is of Array Size 8721. For a single sample arr, this is how we can access our data:
        arr[:7] -> 7 Forcing Variables
        arr[7] -> Month Indicator of Data
        arr[8] -> LST Target
        arr[9:].reshape(7,33,33) -> Sattelite Image Tensor 
        """
        finetune_day_tot_samples = 0
        finetune_night_tot_samples = 0

        finetune_day_first_batch = True
        finetune_night_first_batch = True

        building_height_dataset=pd.read_csv(PATH_TO_BUILDING_HEIGHT)['0']
        elevation_dataset = TFRecordsElevation(filepath=PATH_TO_ELEVATION).tfrecrods2numpy()

        lat_lon=pd.read_csv(PATH_TO_LATLONG)
        total_pixels=lat_lon.shape[0]

        print(f'Storing data at {finetune_day_path},{finetune_night_path}.')
        hp5_finetune_day = h5py.File(finetune_day_path, "a")
        hp5_finetune_night = h5py.File(finetune_night_path, "a")

        with open(PATH_TO_FORCING_DAY, 'rb') as f:
            forcing_data_day = pickle.load(f)
        with open(PATH_TO_FORCING_NIGHT, 'rb') as f:
            forcing_data_night = pickle.load(f)
        print('forcing loaded.')
        
        station_data_day = pd.read_csv(PATH_TO_STATION_DAY)
        station_data_night = pd.read_csv(PATH_TO_STATION_NIGHT)
        idx_all=station_data_day.idx.unique()
        print(f'station loaded. index:{idx_all}')

        avail_dates = list(forcing_data_day.keys())

        files_1 = [os.path.join(PATH_TO_DATA_1, file) for file in os.listdir(PATH_TO_DATA_1) if 'modis' in file]
        files_2 = [os.path.join(PATH_TO_DATA_2, file) for file in os.listdir(PATH_TO_DATA_2) if 'modis' in file]
        files = files_1+files_2
        for file_modis in tqdm(sorted(files), desc="Total Progress"):  
            path_to_tf_modis = file_modis
            path_to_tf_landsat = file_modis.replace("modis","landsat")
            records_modis = TFRecordsParser_Modis(path_to_tf_modis).tfrecrods2numpy()
            records_landsat = TFRecordsParser_Landsat(path_to_tf_landsat, channels=channel_choices).tfrecrods2numpy()
            num_days =int(np.shape(records_modis)[0]/total_pixels)
            print(file_modis,num_days)

            indices_to_delete = []
            for idx in range(len(records_modis)):
                if int(records_modis[idx][0]) not in np.arange(lat_lon.shape[0]):
                    print(idx, records_modis[idx])
                    indices_to_delete.append(idx)
            for idx in indices_to_delete:
                records_modis.pop(idx) 
            
            for day_i in range(num_days):
                finetune_day_samples = []
                finetune_night_samples = []
                file_date = records_modis[day_i*total_pixels][1]
                records_i=records_modis[day_i*total_pixels:(day_i+1)*total_pixels]
                matching_date = "20" + file_date[0:2] + "-" + file_date[2:4] + "-" + file_date[4:]

                if int(records_i[-1][0])!=total_pixels-1:
                    print(int(records_i[-1][0]))
                    print(len(records_modis))
                    records_modis.insert((day_i+1)*total_pixels-1,(False, False, False, False, False, False))
                    print(len(records_modis))
                assert(int(records_i[0][0])==0)
                
                if matching_date in avail_dates:
                    cloud_pixels_day = Counter(elem[2] for elem in records_i)[False]
                    cloud_pct_day = cloud_pixels_day/total_pixels
                    cloud_pixels_night = Counter(elem[4] for elem in records_i)[False]
                    cloud_pct_night = cloud_pixels_night/total_pixels
                    year = int(file_date[0:2])
                    month = int(file_date[2:4])
                    day = int(file_date[4:6])
                    for _, (idx_loc, date, lst_day, QC_day, lst_night, QC_night) in enumerate(records_i):
                        idx=int(idx_loc)
                        features = records_landsat[idx]
                        if ((features>=0).all() == True) and ((features<=1).all() == True) and idx in idx_all:
                            NIR_dn = features[3]
                            SWIR1_dn = features[4]
                            RED_dn = features[0]

                            features_ndbi = ((SWIR1_dn-NIR_dn)/(SWIR1_dn+NIR_dn)).reshape(-1,33,33)
                            features_ndvi = ((NIR_dn-RED_dn)/(NIR_dn+RED_dn)).reshape(-1,33,33)
                            features_elev = elevation_dataset[idx].reshape(-1,33,33)
                            features = np.concatenate([features,features_ndbi,features_ndvi,features_elev],axis=0)
                            forcing_day = forcing_data_day[matching_date][idx]
                            forcing_night = forcing_data_night[matching_date][idx]
                            building_height = np.array(building_height_dataset[idx]).reshape(-1,)

                            sample_image = features.flatten()
                            loc_idx = np.array(int(idx)).reshape(-1,)
                            year = np.array(year).reshape(-1,)
                            month = np.array(month).reshape(-1,)
                            day = np.array(day).reshape(-1,)

                            if ((features_ndvi>1).any() == True) or ((features_ndvi<-1).any() == True):
                                print(f'NDVI values going out of range: {features_ndvi.min()}, {features_ndvi.max()}.')

                            if (QC_day>>1&1)==0 and (QC_day>>7&1)==0 and (lst_day is not False) and random.uniform(0,1) <= THRESH and cloud_pct_day<=threshold_cloud:
                                lst_day = np.array(lst_day).reshape(-1,)
                                sliced_station_data_day = station_data_day.loc[(station_data_day.loc[:, "Date"] == matching_date) & (station_data_day.loc[:, "idx"] == idx)]
                                if len(sliced_station_data_day) > 0:
                                    station_temp_day = np.array(sliced_station_data_day.T_2m.values[0]).reshape(-1,)
                                    station_day_ex_array = np.concatenate((loc_idx, year, month, day, forcing_day, building_height, lst_day, station_temp_day, sample_image))
                                    finetune_day_samples.append(station_day_ex_array)
                            if (QC_night>>1&1)==0 and (QC_night>>7&1)==0 and (lst_night is not False) and random.uniform(0,1) <= THRESH and cloud_pct_night<=threshold_cloud:
                                lst_night = np.array(lst_night).reshape(-1,)
                                sliced_station_data_night = station_data_night.loc[(station_data_night.loc[:, "Date"] == matching_date) & (station_data_night.loc[:, "idx"] == idx)]
                                if len(sliced_station_data_night) > 0:
                                    station_temp_night = np.array(sliced_station_data_night.T_2m.values[0]).reshape(-1,)
                                    station_night_ex_array = np.concatenate((loc_idx, year, month, day, forcing_night, building_height, lst_night, station_temp_night, sample_image))
                                    finetune_night_samples.append(station_night_ex_array)

                finetune_day_samples = np.array(finetune_day_samples)
                finetune_day_new_samples_num = finetune_day_samples.shape[0]
                if finetune_day_new_samples_num > 0:
                    ## ONLY APPEND IF DATA EXISTS ##
                    finetune_day_tot_samples += finetune_day_new_samples_num
                    _, feature_length = finetune_day_samples.shape
                    assert(feature_length == 8727)
                    if finetune_day_first_batch:
                        finetune_day_hdf5_dataset = hp5_finetune_day.create_dataset(city_name, (finetune_day_new_samples_num, feature_length), maxshape=(None, feature_length), dtype='float32')
                        finetune_day_hdf5_dataset[:] = finetune_day_samples
                        finetune_day_first_batch = False

                    finetune_day_hdf5_dataset.resize(finetune_day_tot_samples, axis=0)
                    finetune_day_hdf5_dataset[-finetune_day_new_samples_num:] = finetune_day_samples                    

                finetune_night_samples = np.array(finetune_night_samples)
                finetune_night_new_samples_num = finetune_night_samples.shape[0]
                if finetune_night_new_samples_num > 0:
                    ## ONLY APPEND IF DATA EXISTS ##
                    finetune_night_tot_samples += finetune_night_new_samples_num
                    _, feature_length = finetune_night_samples.shape
                    assert(feature_length == 8727)
                    if finetune_night_first_batch:
                        finetune_night_hdf5_dataset = hp5_finetune_night.create_dataset(city_name, (finetune_night_new_samples_num, feature_length), maxshape=(None, feature_length), dtype='float32')
                        finetune_night_hdf5_dataset[:] = finetune_night_samples
                        finetune_night_first_batch = False

                    finetune_night_hdf5_dataset.resize(finetune_night_tot_samples, axis=0)
                    finetune_night_hdf5_dataset[-finetune_night_new_samples_num:] = finetune_night_samples
                                    
        hp5_finetune_day.close()
        hp5_finetune_night.close()
        return "DONE"
#    try:
    extract_data(PATH_TO_STORE_FINETUNE_DAY,PATH_TO_STORE_FINETUNE_NIGHT)
    hf = h5py.File(PATH_TO_STORE_FINETUNE_DAY, 'r')
    num_samples=hf[city_name].shape[0]
    print(f'Total number of finetune day samples: {num_samples}.')
    hf.close()
    hf = h5py.File(PATH_TO_STORE_FINETUNE_NIGHT, 'r')
    num_samples=hf[city_name].shape[0]
    print(f'Total number of finetune night samples: {num_samples}.')
    hf.close()
#    except:
#        print('City did not work.')
