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

PATH_TO_PICKLE = "data/xarray_data.pkl"
PATH_TO_DATA = "/mnt/sdc2/data/research_data/sat_lst"
PATH_TO_STORE = "/mnt/sdc2/data/research_data/sharded_data/"
PATH_TO_RAND_STORE = "/mnt/sdc2/data/research_data/rand_sharded_data/"
PATH_TO_LATLONG = "/mnt/sdc2/data/research_data/modis_lon_lat.csv"
PATH_TO_ELEVATIONS = "/mnt/sdc2/data/research_data/AWS3D30_cropped.tfrecord"

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
    center_weight = 0.5
    outside_weight = 1 - center_weight
    weights_idx_dict = inverse_distance_mean()
    elevations_dict = TFRecordsElevation(filepath=PATH_TO_ELEVATIONS).tfrecrods2numpy()
    sink = wds.ShardWriter(PATH_TO_STORE+"shard-%06d.tar", maxcount=10000, encoder=True)
    with open(PATH_TO_PICKLE, 'rb') as f:
        forcing_data = pickle.load(f)

    avail_dates = list(forcing_data.keys())

    for root, dirs, files in os.walk(PATH_TO_DATA):
        for file in tqdm(files, desc="Total Progress", position=0):
            path_to_tf = os.path.join(PATH_TO_DATA, file)
            file_date = file.split(".")[0]
            records = TFRecordsParser(path_to_tf).tfrecrods2numpy()

            if file_date in avail_dates:
                for idx, (features, lst) in tqdm(enumerate(records), desc="Storing Files", position=1,
                                                 total=len(records), leave=False):
                    weights = np.array(weights_idx_dict[idx]["weights"]) * outside_weight
                    min_idx = weights_idx_dict[idx]["idx"]
                    surround_features = np.array([records[i][0] for i in min_idx])
                    weighted_avg = np.average(surround_features, weights=weights, axis=0)
                    features = np.average([features, weighted_avg], weights=[center_weight, outside_weight], axis=0)
                    elevations = elevations_dict[idx]
                    # elevations = elevations_dict[idx].clip(0, 500)
                    # scaler = MinMaxScaler()
                    # elevations = np.expand_dims(scaler.fit_transform(elevations), axis=0)
                    features = np.vstack([features, elevations])
                    forcing = forcing_data[file_date][idx]
                    if lst is not False:
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

