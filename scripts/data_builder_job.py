import numpy as np
from tfrecords2numpy import TFRecordsParser
from tfrecords2numpy import TFRecordsElevation
import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import webdataset as wds
import xarray as xr

#PATH_TO_FORCING = "/glade/work/yiwenz/TransferLearning/CESM_0.125_modis_Daily/CESM_0.125_Daily_all.nc"
PATH_TO_FORCING = "/glade/work/yiwenz/TransferLearning/forcing_daily.pkl"
PATH_TO_DATA = "/glade/scratch/yiwenz/Data_TFRecord_Daily/"
PATH_TO_STORE = "/glade/scratch/yiwenz/TransferLearningData/sharded_data_all_Daily_dropna/"
PATH_TO_RAND_STORE = "/glade/scratch/yiwenz/TransferLearningData/rand_sharded_data_all_Daily_dropna/"
PATH_TO_LATLONG = "/glade/u/home/yiwenz/TransferLearning/modis_lon_lat.csv"
PATH_TO_ELEVATIONS = "/glade/work/yiwenz/AWS3D30_cropped.tfrecord"

def inverse_distance_mean():
    latlong = pd.read_csv(PATH_TO_LATLONG, usecols=["lon", "lat"])
    array = latlong.values
    dist = euclidean_distances(array, array)
    weights_idx_dict = {}
    for idx in range(len(dist)):
        sort_idx = dist[idx, :].argsort()[1:8]
        closest = dist[idx, sort_idx]**-1
        closest = closest / closest.sum()
        weights_idx_dict[idx] = {"weights":closest, "idx":sort_idx}

    return weights_idx_dict

def extract_data():
    """
    Extract all data from all TFRecords files and stores as pickled tuples in the format (image, label)
    :return:
    """
    center_weight = 1
    outside_weight = 1 - center_weight
    weights_idx_dict = inverse_distance_mean()
    elevations_dict = TFRecordsElevation(filepath=PATH_TO_ELEVATIONS).tfrecrods2numpy()
    sink = wds.ShardWriter(PATH_TO_STORE+"shard-%06d.tar", maxcount=10000, encoder=True)
    with open(PATH_TO_FORCING, 'rb') as f:
        forcing_data = pickle.load(f)
#    forcing_data = xr.open_dataset(PATH_TO_FORCING)
    
    avail_dates = list(forcing_data.keys())
#    avail_dates=forcing_data.time.dt.strftime('%y%m%d').values.tolist()
    avail_dates = [s.replace("20", "").replace("-","") for s in avail_dates]
#    print(avail_dates)
    
    for root, dirs, files in os.walk(PATH_TO_DATA):
        for file in tqdm(files, desc="Total Progress", position=0):
            path_to_tf = os.path.join(PATH_TO_DATA, file)
            file_date = file.split(".")[0]
            records = TFRecordsParser(path_to_tf).tfrecrods2numpy()
            forcing_date = '20'+file_date[:2]+'-'+file_date[2:4]+'-'+file_date[4:6]
            print(file_date,forcing_date)

            if file_date in avail_dates:
                for idx, (features, lst) in tqdm(enumerate(records), desc="Storing Files", position=1,
                                                 total=len(records), leave=False):
                    if (lst is not False) and ((features!=-9999).all() == True):
                        if outside_weight != 0:
                            weights = np.array(weights_idx_dict[idx]["weights"]) * outside_weight
                            min_idx = weights_idx_dict[idx]["idx"]
                            surround_features = np.array([records[i][0] for i in min_idx])
                            weighted_avg = np.average(surround_features, weights=weights, axis=0)
                            features = np.average([features, weighted_avg], weights=[center_weight, outside_weight], axis=0)
                        NIR_dn = (features[3]+0.2)/2.75e-05
                        SWIR1_dn = (features[4]+0.2)/2.75e-05
                        RED_dn = (features[0]+0.2)/2.75e-05
                        # features_ndbi=np.empty([1,33,33])
                        # features_ndvi=np.empty([1,33,33])
                        # for index, values in np.ndenumerate(features[4]):
                        #     values_NIR = features[3][index]
                        #     if values==-9999 or values_NIR==-9999:
                        #         features_ndbi[0][index] = -9999
                        #     else:
                        #         features_ndbi[0][index] = ((SWIR1_dn[index]-NIR_dn[index])/(SWIR1_dn[index]+NIR_dn[index]))
                        # features_ndbi=features_ndbi.reshape(-1,33,33)
                        # for index, values in np.ndenumerate(features[3]):
                        #     values_Red = features[0][index]
                        #     if values==-9999 or values_Red==-9999:
                        #         features_ndvi[0][index] = -9999
                        #     else:
                        #         features_ndvi[0][index] = ((NIR_dn[index]-RED_dn[index])/(NIR_dn[index]+RED_dn[index]))
                        # features_ndvi=features_ndvi.reshape(-1,33,33)
                        features_ndbi = ((SWIR1_dn-NIR_dn)/(SWIR1_dn+NIR_dn)).reshape(-1,33,33)
                        features_ndvi = ((NIR_dn-RED_dn)/(NIR_dn+RED_dn)).reshape(-1,33,33)
                        features = np.concatenate([features,features_ndbi,features_ndvi],axis=0)
                        elevations = elevations_dict[idx].reshape(-1,33,33)
                        # elevations = elevations_dict[idx].clip(0, 500)
                        # scaler = MinMaxScaler()
                        # elevations = np.expand_dims(scaler.fit_transform(elevations), axis=0)
                        features = np.vstack([features, elevations])
                        forcing = forcing_data[forcing_date][idx]
    #                    date_stamp='20'+file_date[:2]+'-'+file_date[2:4]+'-'+file_date[4:6]
    #                    forcing=forcing_data.sel(time=date_stamp,index=idx).to_array().values      
                        sample = {
                            "__key__": f"{file_date}_{idx}",
                            "image.pyd": features,
                            "forcing.pyd": forcing,
                            "lst.pyd": lst
                        }
                        sink.write(sample)
    sink.close()

def shuffle_data():
    files = []
    for dirpath, dirnames, filenames in os.walk(PATH_TO_STORE):
        files.extend(filenames)
    files = sorted(files)
    files_nums = [file[6:12] for file in files]
    path_to_files = PATH_TO_STORE + "shard-{{{}..{}}}.tar".format(files_nums[0], files_nums[-1])
    dataset = wds.WebDataset(path_to_files).decode("rgb").shuffle(10000, initial=10000)
    sink = wds.ShardWriter(PATH_TO_RAND_STORE + "shard-%06d.tar", maxcount=10000, encoder=True)
    for data in tqdm(dataset, total=len(files_nums)*10000):
        key = data["__key__"]
        forcing = data["forcing.pyd"]
        image = data["image.pyd"]
        lst = data["lst.pyd"]
        sample = {
            "__key__": key,
            "image.pyd": image,
            "forcing.pyd": forcing,
            "lst.pyd": lst
        }
        sink.write(sample)
    sink.close()

if __name__ == "__main__":
    extract_data()
    shuffle_data()
    # import torch
    # path = PATH_TO_STORE+"shard-{000000..000004}.tar"
    # dataset = wds.WebDataset(path).decode("rgb").to_tuple("image.pyd", "forcing.pyd", "lst.pyd")
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1)
    # for idx, data in enumerate(dataloader):
    #     if idx % 1000 == 0:
    #         print(idx)