import h5py
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

country_name="US"

city_identifier = pd.read_csv(f"/glade/u/home/yiwenz/TransferLearning/City_NameList/{country_name}_384.csv").reset_index(drop=False)
PATH_TO_DATA = f"/glade/derecho/scratch/yiwenz/pretrain_AllCountries/pretrain_city_hdf5_night_926_ERA5_US_0-1_all_monthly/" #{country_name}
PATH_TO_STORE = f"/glade/derecho/scratch/yiwenz/pretrain_Shuffled/pretrain_city_hdf5_night_926_ERA5_US_0-1_all_monthly_shuffle"
PATH_TO_DICT= os.path.join(PATH_TO_STORE,"remain_samples.pkl")
pretraining_paths = [os.path.join(PATH_TO_DATA, file) for file in os.listdir(PATH_TO_DATA) if '.h5' in file ]
#pretraining_paths = sorted(pretraining_paths, key = os.path.getsize)[-11:]
print(len(pretraining_paths))
wanted_samples_train=1000000*26
wanted_samples_test=1000000*3
max_file_length=1000000
print(wanted_samples_train,wanted_samples_test)
city_id_dictionary = dict(zip(city_identifier.City_Name.astype('str'), city_identifier.index))
train_pct=0.7

class TrainingDataBuilder:
    def __init__(self, 
                 path_to_h5s,
                 path_to_save_dir,
                 max_file_length, 
                 wanted_samples_train,
                 wanted_samples_test,
                 train_pct,
                 feature_vector_length):
        
        self.path_to_h5s = path_to_h5s
        self.path_to_save_dir = path_to_save_dir
        self.max_file_length = max_file_length
        self.wanted_samples_train = wanted_samples_train
        self.wanted_samples_test = wanted_samples_test
        self.train_pct = train_pct
        self.feature_vector_length = feature_vector_length
        self.length_from_city = int(max_file_length/len(path_to_h5s))
        
        self.load_data()
        print(f"Grabbing {self.length_from_city*self.train_pct} training samples and {self.length_from_city*(1-self.train_pct)} testing samples from each city.")
    

        
    def load_data(self):
        self.data = {}
        for file in self.path_to_h5s:
            print(file)
            f = h5py.File(file, 'r')
            key = list(f.keys())[0]
            city_indicator = city_id_dictionary[key]
            arr = f[key]
            samples_idx = np.arange(arr.shape[0])
            print(f"Total Samples in {key}: {arr.shape[0]}.")

            np.random.shuffle(samples_idx)
            if list(f.keys())[0] not in self.data.keys():
                self.data[list(f.keys())[0]] = {"dataset": f,
                                                "all_index_train": samples_idx[:int(arr.shape[0]*self.train_pct)],
                                                "sample_index_train": samples_idx[:int(arr.shape[0]*self.train_pct)], 
                                                "sample_index_test": samples_idx[int(arr.shape[0]*self.train_pct):], 
                                                "total_samples": len(samples_idx), 
                                                "city_indicator":city_indicator}
        
    def grab_samples(self, dataset, indexes, city_indicator):
        city_indicator = np.array(city_indicator).reshape(1)
        indexes = np.sort(indexes)
        subset = np.zeros((len(indexes), self.feature_vector_length))
        for sample_idx, index in enumerate(indexes):
            sample = dataset[index]
            loc_idx = int(sample[0])
            sample = np.concatenate((city_indicator, sample))
            subset[sample_idx] = sample
        subset = subset[~np.isnan(subset).any(axis=1)]
        return subset
    
    def save_file(self, array, filename):
        path_to_save = os.path.join(self.path_to_save_dir, filename)
        h5f = h5py.File(path_to_save, 'w')
        h5f.create_dataset(filename, data=array, dtype='float32')
        h5f.close()
        
    def build_training_data(self):
#Build training set
        save_dict={}
        for key, value in self.data.items():
            save_dict[key] = {"all_index_train":self.data[key]["all_index_train"],
                        "sample_index_train": self.data[key]["sample_index_train"], 
                        "sample_index_test": self.data[key]["sample_index_test"],
                        "city_indicator":self.data[key]["city_indicator"]}
#Number of samples to grab in each file, repeat n times and store in n files
        samples_per_saved_file_train = [self.max_file_length for i in \
                                  range(int(self.wanted_samples_train / self.max_file_length))]
#Remaining wanted number of samples are stored in another file        
        remainder = int(self.wanted_samples_train) % self.max_file_length
        if remainder != 0:
            samples_per_saved_file_train.append(remainder)
#for each file        
        for idx, num_samples in tqdm(enumerate(samples_per_saved_file_train)):
            print(f"### PREPPING TRAINING DATASET {idx} ###")

            subsets = []
            for key, value in self.data.items():
                sample_index = value["sample_index_train"]
                if len(sample_index) <= self.length_from_city:
                    num_samples_total_to_grab = self.length_from_city                    
                    num_samples_grabbed = len(sample_index)
                    num_samples_to_grab = num_samples_total_to_grab-num_samples_grabbed
                    self.data[key]["sample_index_train"] = self.data[key]["all_index_train"]
                    sample_idx_to_grab = np.concatenate((sample_index,self.data[key]["sample_index_train"][:num_samples_to_grab]))
                    self.data[key]["sample_index_train"] = self.data[key]["sample_index_train"][num_samples_to_grab:]
                    
                    save_dict[key]["sample_index_train"] = self.data[key]["sample_index_train"]
                    subset = self.grab_samples(self.data[key]["dataset"][key], sample_idx_to_grab, value["city_indicator"])
                    subsets.append(subset)  
                    print(f'{key},{num_samples_total_to_grab}')

#else, grab # of samples as indicated (either by percent or a fixed amount)
                else:
                    num_samples_to_grab = self.length_from_city
                    sample_idx_to_grab = sample_index[:num_samples_to_grab]
                    remaining_samples = sample_index[num_samples_to_grab:]
#update sample indexes
                    self.data[key]["sample_index_train"] = remaining_samples
#save remaining indexes to dictionary 
                    save_dict[key]["sample_index_train"] = remaining_samples
                    subset = self.grab_samples(self.data[key]["dataset"][key], sample_idx_to_grab, value["city_indicator"])
                    subsets.append(subset)
                    
                    print(f'{key},{num_samples_to_grab}')

                    
            print("Shuffling Dataset")
            subsets = np.concatenate(subsets, axis=0)
            print("Total Samples", len(subsets))
            np.random.shuffle(subsets)
            print("Saving Dataset")
            filename = f"training_{idx+1}.h5"
            self.save_file(subsets, filename)

            f = open(PATH_TO_DICT,"wb")
            pickle.dump(save_dict,f)
            f.close()

#Build test set
#Number of samples to grab in each file, repeat n times and store in n files
        samples_per_saved_file_test = [self.max_file_length for i in \
                                  range(int(self.wanted_samples_test / self.max_file_length))]
#Remaining wanted number of samples are stored in another file        
        remainder = int(self.wanted_samples_test) % self.max_file_length
        if remainder != 0:
            samples_per_saved_file_test.append(remainder)
#for each file        
        for idx, num_samples in tqdm(enumerate(samples_per_saved_file_test)):
            print(f"### PREPPING TESTING DATASET {idx} ###")

            subsets = []
            for key, value in self.data.items():
                sample_index = value["sample_index_test"]
                if len(sample_index) == 0:
                    continue
#if # of samples left is less than wanted from that city, then grab what's left
                elif len(sample_index) <= self.length_from_city:
                    num_samples_to_grab = len(sample_index)
                    sample_idx_to_grab = sample_index
                    #update sample indexes
                    self.data[key]["sample_index_test"] = []
#save remaining indexes to dictionary
                    save_dict[key]["sample_index_test"] = []
                    
                    subset = self.grab_samples(self.data[key]["dataset"][key], sample_idx_to_grab, value["city_indicator"])
                    subsets.append(subset)

#else, grab # of samples as indicated (either by percent or a fixed amount)
                else:
                    num_samples_to_grab = self.length_from_city
                    sample_idx_to_grab = sample_index[:num_samples_to_grab]
                    remaining_samples = sample_index[num_samples_to_grab:]
#update sample indexes
                    self.data[key]["sample_index_test"] = remaining_samples
#save remaining indexes to dictionary 
                    save_dict[key]["sample_index_test"]= remaining_samples
    
                    subset = self.grab_samples(self.data[key]["dataset"][key], sample_idx_to_grab, value["city_indicator"])
                    subsets.append(subset)
                    
                print(f'{key},{num_samples_to_grab}')

                    
            print("Shuffling Dataset")
            subsets = np.concatenate(subsets, axis=0)
            print("Total Samples", len(subsets))
            np.random.shuffle(subsets)
            print("Saving Dataset")
            filename = f"testing_{idx+1}.h5"
            self.save_file(subsets, filename)

            f = open(PATH_TO_DICT,"wb")
            pickle.dump(save_dict,f)
            f.close()    

builder = TrainingDataBuilder(path_to_h5s=pretraining_paths,
                              path_to_save_dir=PATH_TO_STORE,
                              max_file_length = max_file_length,
                              wanted_samples_train = wanted_samples_train,
                              wanted_samples_test = wanted_samples_test,
                              train_pct=train_pct,
                              feature_vector_length=8727)
builder.build_training_data()
