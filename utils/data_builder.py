import numpy as np
from tfrecords2numpy import TFRecordsParser
import os
import pickle
from tqdm import tqdm
import webdataset as wds

PATH_TO_PICKLE = "data/xarray_data.pkl"
PATH_TO_DATA = "/mnt/sdc2/data/research_data/sat_lst"
PATH_TO_STORE = "/mnt/sdc2/data/research_data/sharded_data/"
def extract_data():
    """
    Extract all data from all TFRecords files and stores as pickled tuples in the format (image, label)
    :return:
    """
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
                    forcing = forcing_data[file_date][idx]
                    if lst is not False:
                        sample = {
                            "__key__": f"file_date_{idx}",
                            "image.pyd": features,
                            "forcing.pyd": forcing,
                            "lst.pyd": lst
                        }

                        sink.write(sample)

    sink.close()
if __name__ == "__main__":
    # extract_data()
    import torch
    path = PATH_TO_STORE+"shard-{000000..000004}.tar"
    dataset = wds.WebDataset(path).decode("rgb").to_tuple("image.pyd", "forcing.pyd", "lst.pyd")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1)
    for idx, data in enumerate(dataloader):
        if idx % 1000 == 0:
            print(idx)

