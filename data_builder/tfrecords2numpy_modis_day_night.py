import tensorflow as tf
import numpy as np
import os

class TFRecordsParser:
    """
    Initial Data Format as a tfrecord. To use with other Deep Learning Frameworks, we can convert to
    numpy arrays for convenience.
    """
    def __init__(self, filepath, label=None):
        # Load in tfrecords and variables
        self.raw_image_dataset = tf.data.TFRecordDataset(filepath)
        self.label = label

        if self.label is None:
            self.label = ["LST_Day_1km",'QC_Day',"LST_Night_1km",'QC_Night','date','id']
#            self.label = "LST_Day_1km"

    def tfrecrods2numpy(self, clip=True):
        records = []
        for _, raw_record in enumerate(self.raw_image_dataset):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            lst_day = np.array(example.features.feature[self.label[0]].float_list.value)
            QC_day = np.array(example.features.feature[self.label[1]].float_list.value)
            lst_night = np.array(example.features.feature[self.label[2]].float_list.value)
            QC_night = np.array(example.features.feature[self.label[3]].float_list.value)
            date = np.array(example.features.feature[self.label[4]].bytes_list.value)[0].decode("utf-8") 
            idx_loc = np.array(example.features.feature[self.label[5]].bytes_list.value)[0].decode("utf-8") 
            if len(lst_day) == 0: # If LST doens't exist then set to false to filter out later
                lst_day = False
            else:
                lst_day = lst_day[0]
                
            if len(QC_day) == 0: # If QC doens't exist then set to false to filter out later
                QC_day = 0
            else:
                QC_day = int(QC_day[0])

            if len(lst_night) == 0: # If LST doens't exist then set to false to filter out later
                lst_night = False
            else:
                lst_night = lst_night[0]
                
            if len(QC_night) == 0: # If QC doens't exist then set to false to filter out later
                QC_night = 0
            else:
                QC_night = int(QC_night[0])

            records.append((idx_loc, date, lst_day, QC_day, lst_night, QC_night))
        return records


class TFRecordsElevation:
    def __init__(self, filepath, channels=None, label=None, image_dim=(33,33)):
        self.raw_elevation_dataset = tf.data.TFRecordDataset(filepath)
        self.channels = channels
        self.image_dim = image_dim
        self.label = label

        self.channels = "elevation"

    def tfrecrods2numpy(self, clip=True):
        elevations = {}
        for idx, raw_record in enumerate(self.raw_elevation_dataset):
            feature = self.channels
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            elevations[idx] = np.array(example.features.feature[feature].float_list.value).reshape(self.image_dim)

        return elevations

if __name__ == "__main__":
    tfe = TFRecordsElevation(filepath)
    tfe.tfrecrods2numpy()
