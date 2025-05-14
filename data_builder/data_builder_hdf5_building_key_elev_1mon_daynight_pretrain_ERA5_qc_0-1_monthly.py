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


# city_idx_start,city_idx_end=[320,321]
city_idx_all=[52,72,118,153,176,184,318,325]

threshold_cloud=0.3
THRESH=1
city_400_file='/glade/u/home/yiwenz/TransferLearning/City_NameList/US_384.csv'
cities_400=pd.read_csv(city_400_file)[['City_Name','NAME10','Time_zone']]

# for city_idx in range(city_idx_start,city_idx_end):
for city_idx in city_idx_all: 
        
    city_list=cities_400.loc[city_idx].to_list()
    city_name, _, _=city_list
    print(city_name)       
    PATH_TO_DATA_1 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite'
    PATH_TO_DATA_2 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US_2019_2024/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite_2019_2024'
    PATH_TO_FORCING_DAY = "/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Day_Regrid_US_All/forcing_"+city_name+".pkl"
    PATH_TO_FORCING_NIGHT = "/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Night_Regrid_US_All/forcing_"+city_name+".pkl" 
    PATH_TO_STORE_PRETRAIN_DAY = f"/glade/derecho/scratch/yiwenz/pretrain_AllCountries/pretrain_city_hdf5_day_926_ERA5_US_0-1_all_monthly/{city_name}_{threshold_cloud}_landsat8_pretrain_day.h5"
    PATH_TO_STORE_PRETRAIN_NIGHT = f"/glade/derecho/scratch/yiwenz/pretrain_AllCountries/pretrain_city_hdf5_night_926_ERA5_US_0-1_all_monthly/{city_name}_{threshold_cloud}_landsat8_pretrain_night.h5" 
    
    PATH_TO_LATLONG = "/glade/work/yiwenz/modis_lat_lon_US_926/modis_grid_"+city_name+".csv"
    PATH_TO_BUILDING_HEIGHT = "/glade/work/yiwenz/building_height_US/building_height_"+city_name+".csv"
    PATH_TO_ELEVATION = "/glade/work/yiwenz/elevation_US/"+city_name+".tfrecord"
    channel_choices = ['Red', 'Green', 'Blue', "NIR", "SWIR1"]

    def extract_data(pretrain_day_path,pretrain_night_path):
        """
        Extract all data from all TFRecords files and stores as pickled tuples in the format (image, label)
        :return:
        Each Sample is of Array Size 8721. For a single sample arr, this is how we can access our data:
        arr[:7] -> 7 Forcing Variables
        arr[7] -> Month Indicator of Data
        arr[8] -> LST Target
        arr[9:].reshape(7,33,33) -> Sattelite Image Tensor 
        """
        pretrain_day_tot_samples = []
        pretrain_night_tot_samples = []

        pretrain_day_first_batch = True
        pretrain_night_first_batch = True

        avail_days_day=0
        avail_days_night=0

        building_height_dataset=pd.read_csv(PATH_TO_BUILDING_HEIGHT)['0']
        elevation_dataset = TFRecordsElevation(filepath=PATH_TO_ELEVATION).tfrecrods2numpy()

        lat_lon=pd.read_csv(PATH_TO_LATLONG)
        total_pixels=lat_lon.shape[0]

        print(f'Storing data at {pretrain_day_path},{pretrain_night_path}.')
        hp5_pretrain_day = h5py.File(pretrain_day_path, "a")
        hp5_pretrain_night = h5py.File(pretrain_night_path, "a")

        with open(PATH_TO_FORCING_DAY, 'rb') as f:
            forcing_data_day = pickle.load(f)
        with open(PATH_TO_FORCING_NIGHT, 'rb') as f:
            forcing_data_night = pickle.load(f)
        print('forcing loaded.')

        avail_dates = list(forcing_data_day.keys())

        files_1 = [os.path.join(PATH_TO_DATA_1, file) for file in os.listdir(PATH_TO_DATA_1) if 'modis' in file]
        files_2 = [os.path.join(PATH_TO_DATA_2, file) for file in os.listdir(PATH_TO_DATA_2) if 'modis' in file]
        files = files_1+files_2
#            random.shuffle(files)
        for file_modis in tqdm(sorted(files), desc="Total Progress"):    
            path_to_tf_modis = file_modis
            path_to_tf_landsat = file_modis.replace("modis","landsat")
            if os.path.exists(path_to_tf_landsat):
                records_modis = TFRecordsParser_Modis(path_to_tf_modis).tfrecrods2numpy()
                records_landsat = TFRecordsParser_Landsat(path_to_tf_landsat, channels=channel_choices).tfrecrods2numpy()
                num_days =int(np.shape(records_modis)[0]/total_pixels)
    
                indices_to_delete = []
                for idx in range(len(records_modis)):
                    if int(records_modis[idx][0]) not in np.arange(lat_lon.shape[0]):
                        print(idx, records_modis[idx])
                        indices_to_delete.append(idx)
                for idx in indices_to_delete:
                    records_modis.pop(idx) 
    
                for day_i in range(num_days):
                    pretrain_day_samples = []
                    pretrain_night_samples = []
                    file_date = records_modis[day_i*total_pixels][1]
                    matching_date = "20" + file_date[0:2] + "-" + file_date[2:4] + "-" + file_date[4:]
                    records_i=records_modis[day_i*total_pixels:(day_i+1)*total_pixels]
                    
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
    #                        print(f'Cloud cover pct day: {cloud_pct_day}; Cloud cover pct night: {cloud_pct_night}')
                        if cloud_pct_day<=threshold_cloud:
                            avail_days_day+=1
                        if cloud_pct_night<=threshold_cloud:
                            avail_days_night+=1
                        for _, (idx_loc, date, lst_day, QC_day, lst_night, QC_night) in enumerate(records_i):
                            idx=int(idx_loc)
                            features = records_landsat[idx]
                            if ((features>=0).all() == True) and ((features<=1).all() == True):
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
                                    pretrain_day_ex_array = np.concatenate((loc_idx, year, month, day, forcing_day, building_height, lst_day, sample_image))
                                    pretrain_day_samples.append(pretrain_day_ex_array)
                                if (QC_night>>1&1)==0 and (QC_night>>7&1)==0 and (lst_night is not False) and random.uniform(0,1) <= THRESH and cloud_pct_night<=threshold_cloud:
                                    lst_night = np.array(lst_night).reshape(-1,)
                                    pretrain_night_ex_array = np.concatenate((loc_idx, year, month, day, forcing_night, building_height, lst_night, sample_image))
                                    pretrain_night_samples.append(pretrain_night_ex_array)
    
                    pretrain_day_samples = np.array(pretrain_day_samples)
                    pretrain_day_new_samples_num = pretrain_day_samples.shape[0]
                    if pretrain_day_new_samples_num > 0:
                        pretrain_day_tot_samples.append(pretrain_day_samples)
    
                    pretrain_night_samples = np.array(pretrain_night_samples)
                    pretrain_night_new_samples_num = pretrain_night_samples.shape[0]
                    if pretrain_night_new_samples_num > 0:
                        ## ONLY APPEND IF DATA EXISTS ##
                        pretrain_night_tot_samples.append(pretrain_night_samples)

# Use numpy.append is really slow, so instead append to a list and then use vstack to turn a list of np.arrays into one np.array
        try:
            pretrain_day_tot_samples = np.vstack(pretrain_day_tot_samples)
            f_pd_day=pd.DataFrame(pretrain_day_tot_samples)
            np_group_day=f_pd_day.groupby(by=[0,1,2]).mean().reset_index().to_numpy(dtype='float32')
            np_group_num_day = np_group_day.shape[0]
            hp5_pretrain_day = h5py.File(pretrain_day_path, "a")
            pretrain_day_hdf5_dataset = hp5_pretrain_day.create_dataset(city_name, (np_group_num_day, 8726), maxshape=(None, 8726), dtype='float32')
            pretrain_day_hdf5_dataset[:] = np_group_day
            hp5_pretrain_day.close()      
    
            pretrain_night_tot_samples = np.vstack(pretrain_night_tot_samples)
            f_pd_night=pd.DataFrame(pretrain_night_tot_samples)
            np_group_night=f_pd_night.groupby(by=[0,1,2]).mean().reset_index().to_numpy(dtype='float32')
            np_group_num_night = np_group_night.shape[0]
            hp5_pretrain_night = h5py.File(pretrain_night_path, "a")
            pretrain_night_hdf5_dataset = hp5_pretrain_night.create_dataset(city_name, (np_group_num_night, 8726), maxshape=(None, 8726), dtype='float32')
            pretrain_night_hdf5_dataset[:] = np_group_night
            hp5_pretrain_night.close() 

            print(f'Total pretrain day samples: {f_pd_day.shape[0]}, {avail_days_day} days of data meets the diurnal cloud threshold.')
            print(f'Total pretrain night samples: {f_pd_night.shape[0]}, {avail_days_night} days of data meets the nocturnal cloud threshold.')
        except:
            print('No available samples')
                        
        return "DONE"

    if os.path.exists(PATH_TO_STORE_PRETRAIN_DAY):
        os.remove(PATH_TO_STORE_PRETRAIN_DAY)
    if os.path.exists(PATH_TO_STORE_PRETRAIN_NIGHT):
        os.remove(PATH_TO_STORE_PRETRAIN_NIGHT)
        
    extract_data(PATH_TO_STORE_PRETRAIN_DAY,PATH_TO_STORE_PRETRAIN_NIGHT)
    
    try:
        hf = h5py.File(PATH_TO_STORE_PRETRAIN_DAY, 'r')
        num_samples=hf[city_name].shape[0]
        print(f'Total number of pretraining day samples: {num_samples}.')
        hf.close()
        hf = h5py.File(PATH_TO_STORE_PRETRAIN_NIGHT, 'r')
        num_samples=hf[city_name].shape[0]
        print(f'Total number of pretraining night samples: {num_samples}.')
        hf.close()
    except:
        print("City did not work.")
