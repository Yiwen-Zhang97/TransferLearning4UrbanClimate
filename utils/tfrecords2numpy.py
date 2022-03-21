import tensorflow as tf
import numpy as np
import os

class TFRecordsParser:
    """
    Initial Data Format as a tfrecord. To use with other Deep Learning Frameworks, we can convert to
    numpy arrays for convenience.
    """
    def __init__(self, filepath, channels=None, label=None, image_dim=(33,33)):
        # Load in tfrecords and variables
        self.raw_image_dataset = tf.data.TFRecordDataset(filepath)
        self.channels = channels
        self.image_dim = image_dim
        self.label = label

        # Get which channels we want to extract
        if self.channels is None:
            self.channels = ['Red', 'Green', 'Blue', "NIR"]

        if self.label is None:
            self.label = "LST_Day_1km"

    def tfrecrods2numpy(self, clip=True):
        records = []
        for idx, raw_record in enumerate(self.raw_image_dataset):
            featureset = {}
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for feature in self.channels:
                featureset[feature] = np.array(example.features.feature[feature].float_list.value).reshape(self.image_dim)

            lst = np.array(example.features.feature[self.label].float_list.value)
            if len(lst) == 0: # If LST doens't exist then set to false to filter out later
                lst = False
            else:
                lst = lst[0]

            # Build array out of featureset
            featureset = np.array(list(featureset.values()))
            records.append((featureset, lst))

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
    tfe = TFRecordsElevation(filepath="/mnt/sdc2/data/research_data/AWS3D30_cropped.tfrecord")
    tfe.tfrecrods2numpy()