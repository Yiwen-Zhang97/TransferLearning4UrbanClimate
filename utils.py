import tensorflow as tf
import numpy as np

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
            self.channels = ['Red', 'Green', 'Blue']

        if self.label is None:
            self.label = "LST_Day_1km"

        # Append on label to channels to get full features list
        self.all_features = self.channels + [self.label]

        # Calculate number of pixels per image
        self.num_pixels = image_dim[0] * image_dim[1]

        # Build image feature dictionary
        self.image_feature_description = {}
        for channel in self.channels:
            # Setup parser for image channels
            self.image_feature_description[channel] = tf.io.FixedLenFeature([self.num_pixels], tf.float32)

        # Setup parser for image labels
        self.image_feature_description[self.label] = tf.io.FixedLenFeature([], tf.float32)

    def _parse_image_function(self, input_dataset):
        """
        Tensorflow helper function to parse single images from dataset
        """
        return tf.io.parse_single_example(input_dataset, self.image_feature_description)

    def tfrecrods2numpy(self, clip=True):
        parsed_image_dataset = self.raw_image_dataset.map(self._parse_image_function)

        # Calculate total images in tensorflow dataset
        total_images = 0
        for i in parsed_image_dataset:
            total_images += 1

        # Loop through parsed images and store training data and labels
        index = 0
        num_channels = len(self.channels)
        training_data = np.zeros((total_images, num_channels) + self.image_dim)
        labels = np.zeros((total_images))

        for data in parsed_image_dataset:
            # Create list to store channel data
            channel_data = []
            # Loop through channels and append to list
            for channel in self.channels:
                channel_data.append(data[channel].numpy())

            # Convert list of np arrays to a single np array
            image = np.array(channel_data)
            # Reshape to image dimensions
            image = image.reshape((-1, ) + self.image_dim)

            # Pull label
            label = data[self.label]

            # Clip data to [0,1] if true
            if clip is True:
                image = image.clip(0, 1)

            training_data[index] = image
            labels[index] = label

            index += 1


        return training_data, labels




