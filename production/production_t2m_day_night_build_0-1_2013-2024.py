#single city without index
import numpy as np
import webdataset as wds
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms 
import os
import random
from itertools import islice
import pickle
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import datetime
from tfrecords2numpy_landsat_no_idx import TFRecordsParser as TFRecordsParser_Landsat
from tfrecords2numpy_landsat_no_idx import TFRecordsElevation
from tfrecords2numpy_modis_day_night import TFRecordsParser as TFRecordsParser_Modis
from production_helper import BasicBlock, ResNet_Day, ResNet_Night, BuildPreTrainDataLoaders

city_idx_start,city_idx_end=[226,236]
name_suffix=''
#city_idx_all=[383]
city_400_file='/glade/u/home/yiwenz/TransferLearning/City_NameList/US_384.csv'
cities_400=pd.read_csv(city_400_file)[['City_Name','NAME10','Time_zone']]

PATH_TO_PRED_ROOT = '/glade/u/home/yiwenz/TransferLearning/New_Organized_Code/Train_test_results_model/'
PATH_TO_MODEL_DAY = os.path.join(PATH_TO_PRED_ROOT,'best_resnet_finetune_1_US_ERA5_day_scratchAllFC_nofreeze_moreDrop0.05_lessParam_skipcon_lst_2013-2024_production.pt')
PATH_TO_MODEL_NIGHT = os.path.join(PATH_TO_PRED_ROOT,'best_resnet_finetune_1_US_ERA5_night_scratchLastFC_nofreeze_dropout0.1_skipcon_lst_2013-2024_production.pt')
BATCH_SIZE = 8192
PATH_TO_RESULT_ROOT = '/glade/derecho/scratch/yiwenz/production_Results/production_results_US_lessParam_skipcon_lst_0-1_2013-2024/t2m_city'
PATH_TO_FORCING_DAY_ROOT = '/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Day_Regrid_US_All'
PATH_TO_FORCING_NIGHT_ROOT = '/glade/derecho/scratch/yiwenz/ERA5_Regrid_AllCountries/ERA5_Night_Regrid_US_All'
PATH_TO_LAT_LONG_ROOT = '/glade/work/yiwenz/modis_lat_lon_US_926/'
PATH_TO_BUILDING_HEIGHT_ROOT = '/glade/work/yiwenz/building_height_US/'
PATH_TO_ELEVATION_ROOT = '/glade/work/yiwenz/elevation_US/'
channel_choices = ['Red', 'Green', 'Blue', "NIR", "SWIR1"]
if not os.path.exists(PATH_TO_RESULT_ROOT):
    os.makedirs(PATH_TO_RESULT_ROOT)

DEVICE = "cpu"
#7 channels: ['Red', 'Green', 'Blue', "NIR", "SWIR1","ndbi","ndvi","elevation"]
image_normalize = transforms.Normalize(
                  mean=[0, 0, 0, 0, 0, 0, 0, 2.5839e+02],
                  std=[1, 1, 1, 1, 1, 1, 1, 3.3506e+02]
)

#forcing: ["ssrd", "strd", "tp", "sp", "rh", "temperature", "uwind","vwind","building height"]
#Added building height in forcing
forcing_mean_day = torch.from_numpy(np.array([2.7475e+06, 1.1216e+06, 6.3862e-04, 9.7475e+04, 3.8102e+01, 2.9312e+02, 9.8286e-01, 2.6067e-01, 6.6253e+00]))
forcing_std_day = torch.from_numpy(np.array([6.3123e+05, 2.1583e+05, 3.1177e-03, 5.1533e+03, 1.3052e+01, 1.0468e+01, 2.4544e+00, 2.6655e+00, 1.8508e+00]))
lst_mean_day = torch.from_numpy(np.array([301.1683]))
lst_std_day = torch.from_numpy(np.array([10.3701]))

forcing_mean_night = torch.from_numpy(np.array([1.1066e+06,  5.6395e-04,  9.8393e+04,  6.8529e+01, 2.8995e+02,  3.8538e-01,  1.6821e-01,  6.2637e+00]))
forcing_std_night = torch.from_numpy(np.array([2.1263e+05, 2.0910e-03, 4.2614e+03, 1.7899e+01, 9.5041e+00, 2.7570e+00, 3.0576e+00, 2.2420e+00]))
lst_mean_night = torch.from_numpy(np.array([283.8263]))
lst_std_night = torch.from_numpy(np.array([9.3311]))

lst_mean_day, lst_std_day, lst_mean_night, lst_std_night = lst_mean_day.to(DEVICE), lst_std_day.to(DEVICE), lst_mean_night.to(DEVICE), lst_std_night.to(DEVICE)
forcing_mean_day, forcing_std_day, forcing_mean_night, forcing_std_night  = forcing_mean_day.to(DEVICE), forcing_std_day.to(DEVICE), forcing_mean_night.to(DEVICE), forcing_std_night.to(DEVICE)

def process_data(forcing_day, forcing_night, lst_day, lst_night, image, month):
    image, forcing_day, lst_day,forcing_night,  lst_night= image.to(DEVICE).to(torch.float32), forcing_day.to(DEVICE), lst_day.to(DEVICE), forcing_night.to(DEVICE), lst_night.to(DEVICE)
    month = month-1
    month = month.to(DEVICE).to(torch.int64)
    # Image Transformations
    image[:,:5,] = torch.clip(image[:,:5,], min=0, max=1)
    image[:,5:7,] = torch.clip(image[:,5:7,], min=-1, max=1)
    image = image_normalize(image)
    # Forcing Transformation
    forcing_day = torch.div(torch.sub(forcing_day, forcing_mean_day), forcing_std_day).to(torch.float32)
    forcing_night = forcing_night[:,1:]
    forcing_night = torch.div(torch.sub(forcing_night, forcing_mean_night), forcing_std_night).to(torch.float32)
    # LST Transformation
    lst_day = torch.div(torch.sub(lst_day, lst_mean_day), lst_std_day).to(torch.float32).view(-1, 1)
    lst_night = torch.div(torch.sub(lst_night, lst_mean_night), lst_std_night).to(torch.float32).view(-1, 1)
    return forcing_day, forcing_night, lst_day, lst_night, image, month

random.seed(42)

def resnet_simplified_day():
    return ResNet_Day(BasicBlock,[3,3,0,0])

def resnet_simplified_night():
    return ResNet_Night(BasicBlock,[3,3,0,0])

def load_pretrained_weights(model,path):
    new_state_dict = OrderedDict()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"Loaded pretrained weights from {path}.")

for city_idx in range(city_idx_start,city_idx_end):#city_idx_all:
    city_list=cities_400.loc[city_idx].to_list()
    city_name, _, _=city_list
    print(city_name)
#Define file paths    
    PATH_TO_DATA_1 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite'
    PATH_TO_DATA_2 = "/glade/derecho/scratch/yiwenz/Data_TFRecord_AllCountries/Data_TFRecord_926_Composite_US_2019_2024/Data_TFRecord_Daily_"+city_name+'_Landsat8_Day_Night_926_Composite_2019_2024'
    PATH_TO_FORCING_DAY = os.path.join(PATH_TO_FORCING_DAY_ROOT, f'forcing_{city_name}.pkl')
    PATH_TO_FORCING_NIGHT = os.path.join(PATH_TO_FORCING_NIGHT_ROOT, f'forcing_{city_name}.pkl')
    PATH_TO_LAT_LONG = os.path.join(PATH_TO_LAT_LONG_ROOT, f'modis_grid_{city_name}.csv')
    PATH_TO_BUILDING_HEIGHT = os.path.join(PATH_TO_BUILDING_HEIGHT_ROOT, f'building_height_{city_name}.csv')
    PATH_TO_ELEVATION = os.path.join(PATH_TO_ELEVATION_ROOT, f'{city_name}.tfrecord')
    PATH_TO_RESULT = os.path.join(PATH_TO_RESULT_ROOT, f't2m_prediction_{city_name}{name_suffix}.csv')
#Load predictors
    building_height_dataset=pd.read_csv(PATH_TO_BUILDING_HEIGHT)['0']
    elevation_dataset = TFRecordsElevation(filepath=PATH_TO_ELEVATION).tfrecrods2numpy()
    lat_lon=pd.read_csv(PATH_TO_LAT_LONG)
    total_pixels=lat_lon.shape[0]
    
    with open(PATH_TO_FORCING_DAY, 'rb') as f:
        forcing_data_day = pickle.load(f)
    with open(PATH_TO_FORCING_NIGHT, 'rb') as f:
        forcing_data_night = pickle.load(f)
    print('forcing loaded.')
    avail_dates = list(forcing_data_day.keys())

    files_1 = [os.path.join(PATH_TO_DATA_1, file) for file in os.listdir(PATH_TO_DATA_1) if 'modis' in file]
    files_2 = [os.path.join(PATH_TO_DATA_2, file) for file in os.listdir(PATH_TO_DATA_2) if 'modis' in file]
    files = files_1+files_2
#Load model    
    model_day=resnet_simplified_day()
    load_pretrained_weights(model_day,PATH_TO_MODEL_DAY)
    model_day = model_day.to(DEVICE)
    model_day.eval()
    model_night=resnet_simplified_night()
    load_pretrained_weights(model_night,PATH_TO_MODEL_NIGHT)
    model_night = model_night.to(DEVICE)
    model_night.eval()

    pred_all_day,lst_all_day, pred_all_night,lst_all_night, time_all,idx_all, ndvi_all, ndbi_all, qc_day_all, qc_night_all=[],[],[],[],[],[],[],[],[],[]
    
    with torch.no_grad():
        for file_modis in tqdm(sorted(files), desc="Total Progress"):    
            path_to_tf_modis = file_modis
            path_to_tf_landsat = file_modis.replace("modis","landsat")
            
            if os.path.exists(path_to_tf_landsat):
                records_modis = TFRecordsParser_Modis(path_to_tf_modis).tfrecrods2numpy()
                records_landsat = TFRecordsParser_Landsat(path_to_tf_landsat, channels=channel_choices).tfrecrods2numpy()
                num_days =int(np.shape(records_modis)[0]/total_pixels)
                # print(file_modis,file_landsat,num_days)
    
                indices_to_delete = []
                for idx in range(len(records_modis)):
                    if int(records_modis[idx][0]) not in np.arange(lat_lon.shape[0]):
                        print(idx, records_modis[idx])
                        indices_to_delete.append(idx)
                for idx in indices_to_delete:
                    records_modis.pop(idx) 
                
                production_samples = []
                for day_i in range(num_days):
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
                        year = int(file_date[0:2])
                        month = int(file_date[2:4])
                        day = int(file_date[4:6])
                        
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
                                lst_day = np.array(lst_day).reshape(-1,)
                                lst_night = np.array(lst_night).reshape(-1,)
                                QC_day = np.array(int(QC_day)).reshape(-1,)
                                QC_night = np.array(int(QC_night)).reshape(-1,)
                                loc_idx = np.array(int(idx)).reshape(-1,)
                                year = np.array(year).reshape(-1,)
                                month = np.array(month).reshape(-1,)
                                day = np.array(day).reshape(-1,)
                                mean_ndbi = features_ndbi.mean().reshape(-1,)
                                mean_ndvi = features_ndvi.mean().reshape(-1,)
    
                                production_ex_array = np.concatenate((loc_idx, year, month, day, forcing_day,forcing_night, building_height, lst_day, lst_night, QC_day, QC_night, mean_ndbi, mean_ndvi, sample_image))
                                production_samples.append(production_ex_array)
                production_samples = np.array(production_samples)
                pretrain_trainloader, _, len_train, _ = BuildPreTrainDataLoaders(path_to_data=production_samples,
                                                                                     batch_size=BATCH_SIZE, 
                                                                                     train_pct=1)
    
                for idx, (loc, year, month, day, forcing_day, forcing_night, lst_day, lst_night, QC_day, QC_night, mean_ndbi, mean_ndvi, image) in enumerate(pretrain_trainloader):
                    lst_day=lst_day.to(DEVICE).to(torch.float32)
                    lst_true_day = lst_day.reshape(-1)
                    lst_true_day[lst_true_day == 0] = np.nan
                    lst_true_day = lst_true_day.tolist()
                    
                    lst_night=lst_night.to(DEVICE).to(torch.float32)
                    lst_true_night = lst_night.reshape(-1)
                    lst_true_night[lst_true_night == 0] = np.nan
                    lst_true_night = lst_true_night.tolist()
                    
                    forcing_day, forcing_night, lst_day, lst_night, image, month = process_data(forcing_day, forcing_night, lst_day, lst_night, image, month)
                    one_hot_mon = F.one_hot(month, num_classes=12)
                    
                    pred_day = model_day.forward(image, forcing_day, one_hot_mon,lst_day)
                    pred_day=torch.add(torch.multiply(pred_day,lst_std_day),lst_mean_day)
                    pred_day = pred_day.numpy().reshape(-1).tolist()
                    pred_night = model_night.forward(image, forcing_night, one_hot_mon,lst_night)
                    pred_night=torch.add(torch.multiply(pred_night,lst_std_night),lst_mean_night)
                    pred_night = pred_night.numpy().reshape(-1).tolist()
                    
                    loc_idx = [int(location.item()) for location in loc]
                    yr = [str(int(yy.item())) for yy in year]
                    yr =[('0'+yy) if len(yy)<2 else yy for yy in yr]
                    mon = [str(int(mm.item()+1)) for mm in month]
                    mon =[('0'+mm) if len(mm)<2 else mm for mm in mon]
                    day = [str(int(dd.item())) for dd in day]
                    day =[('0'+dd) if len(dd)<2 else dd for dd in day]
                    time =[(yy+mm+dd) for yy,mm,dd in zip(yr,mon,day)]
    
                    QC_day = QC_day.reshape(-1).tolist()
                    QC_night = QC_night.reshape(-1).tolist()
                    mean_ndbi = mean_ndbi.reshape(-1).tolist()
                    mean_ndvi = mean_ndvi.reshape(-1).tolist()
                    
                    pred_all_day.extend(pred_day)
                    lst_all_day.extend(lst_true_day)
                    pred_all_night.extend(pred_night)
                    lst_all_night.extend(lst_true_night)                
                    time_all.extend(time)
                    idx_all.extend(loc_idx)
                    ndvi_all.extend(mean_ndvi)
                    ndbi_all.extend(mean_ndbi)
                    qc_day_all.extend(QC_day)
                    qc_night_all.extend(QC_night)

    data_dic = {'time':time_all, 'idx':idx_all, 't2m_pred_day': pred_all_day, 'lst_truth_day': lst_all_day, 't2m_pred_night': pred_all_night, 'lst_truth_night': lst_all_night,'QC_day':qc_day_all,'QC_night':qc_night_all, 'ndvi':ndvi_all, 'ndbi':ndbi_all}
    df_data=pd.DataFrame(data=data_dic)
    df_data['time']=df_data.apply(lambda row:datetime.datetime.strptime(row['time'],"%y%m%d"),axis=1)
    df_data[['time', 'idx','t2m_pred_day','lst_truth_day', 't2m_pred_night', 'lst_truth_night','QC_day','QC_night','ndvi','ndbi']].to_csv(PATH_TO_RESULT,index=False)
    print(f'Saved prediction at {PATH_TO_RESULT}.')
