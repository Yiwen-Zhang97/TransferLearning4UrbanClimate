import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
from tqdm import tqdm
import pandas as pd


class PreTrainLoader(Dataset):
    def __init__(self, path):
        self.hf = h5py.File(path, 'r')
        self.key = list(self.hf.keys())[0]
        
        
    def __len__(self):
        return self.hf[self.key].shape[0]
    
    def __getitem__(self, index):
        sample = self.hf[self.key][index, :]
        forcing = sample[5:14]
        month = sample[3]
        lst = sample[14]
        image = sample[15:].reshape(8,33,33)
        return forcing, image, month, lst


class FineTuneLoader(Dataset):
    def __init__(self, dataset, cities=None, sites=None):
        self.dataset = dataset
        self.cities = cities
        self.sites = sites
        
        if self.cities is not None:
            self.dataset = self.dataset [self.dataset['city'].isin(self.cities)]

        if self.sites is not None:
            self.dataset = self.dataset [self.dataset['loc_idx'].astype('int').isin(self.sites)]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset.iloc[index, :].values
        forcing = sample[5:14].astype('float32')
        month = sample[3].astype('float32')
        lst = sample[14].astype('float32')
        t2m = sample[15].astype('float32')
        image = sample[16:].reshape(8,33,33).astype('float32')
        return forcing, image, month, t2m,lst


def BuildPreTrainDataLoaders(path_to_data_dir="/glade/scratch/yiwenz/pretrain_hdf5", 
                     number_hdfs_wanted_train=None, 
                     number_hdfs_wanted_test=None, 
                     batch_size=1024,
                     train_pct=0.7):
    ### LOAD PATHS TO DATASET ###
    path_to_datasets_train = [os.path.join(path_to_data_dir, file) for file in os.listdir(path_to_data_dir) if '.h5' in file and 'training' in file]
    path_to_datasets_train=sorted(path_to_datasets_train, key=lambda x:int(x.partition('training_')[2].partition('.')[0]))
    path_to_datasets_test = [os.path.join(path_to_data_dir, file) for file in os.listdir(path_to_data_dir) if '.h5' in file and 'testing' in file]
    path_to_datasets_test=sorted(path_to_datasets_test, key=lambda x:int(x.partition('testing_')[2].partition('.')[0]))
    
    if number_hdfs_wanted_train is not None:
        path_to_datasets_train = path_to_datasets_train[:number_hdfs_wanted_train]
    if number_hdfs_wanted_test is not None:
        path_to_datasets_test = path_to_datasets_test[:number_hdfs_wanted_test]
        
    ### LOAD DATASETS ###
    datasets_train = [PreTrainLoader(path_to_data) for path_to_data in path_to_datasets_train]
    train_data = torch.utils.data.ConcatDataset(datasets_train)
    datasets_test = [PreTrainLoader(path_to_data) for path_to_data in path_to_datasets_test]
    test_data = torch.utils.data.ConcatDataset(datasets_test)
    
    print("Total Samples for training", len(train_data))
    print("Total Samples for testing", len(test_data))

    ### SEPARATE INTO TRAIN AND TEST ###
#    train_data = torch.utils.data.Subset(trainingset, range(int(len(trainingset) * train_pct)))
#    test_data = torch.utils.data.Subset(trainingset, range(int(len(trainingset) * train_pct), len(trainingset)))

    len_train_data=len(train_data)
    len_test_data=len(test_data)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader, len_train_data, len_test_data

def BuildFineTuneDataLoaders(path_to_data_dir, 
                     batch_size=32,
                     train_pct=0.7,
                     random=True,
                     cities_train=None,
                     cities_test=None,
                     sites_train=None,
                     sites_test=None,
                    sites_val=None):
    ### LOAD PANDAS DATAFRAME ###
    print("Loading Dataset")
    dataset_all = pd.read_csv(path_to_data_dir, index_col=0)
    dataset_all = dataset_all[dataset_all['city'].astype('int')!=0]
    
    ### SEPARATE INTO TRAIN AND TEST ###
    if random == True:
        if sites_train==None:
            dataset = dataset_all[~dataset_all['loc_idx'].astype('int').isin(sites_val)]
        else:
            dataset = dataset_all[dataset_all['loc_idx'].astype('int').isin(sites_train)]
            dataset = dataset[~dataset['loc_idx'].astype('int').isin(sites_val)]
        dataset = FineTuneLoader(dataset=dataset, cities=None)
        train_data = torch.utils.data.Subset(dataset, range(int(len(dataset) * train_pct)))
        test_data = torch.utils.data.Subset(dataset, range(int(len(dataset) * train_pct), len(dataset)))
        
        val_dataset = dataset_all[dataset_all['loc_idx'].astype('int').isin(sites_val)]
        val_data = FineTuneLoader(dataset=val_dataset, cities=None)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif random == False:
        train_data = FineTuneLoader(dataset=dataset, cities=cities_train, sites= sites_train) 
        test_data = FineTuneLoader(dataset=dataset, cities=cities_test, sites= sites_test) 
    
    len_train_data=len(train_data)
    len_test_data=len(test_data)
    len_val_data=len(val_data)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print("DONE")
    return trainloader, testloader, len_train_data, len_test_data,valloader,len_val_data
