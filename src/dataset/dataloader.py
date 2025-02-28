import random
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataset.datareader import get_data
import GPUtil

class DataGenerator(object):
    """
    Data Generator capable of generating batches of data.
    """

    def __init__(self, crop_factor=0.5, num_exposures=3, split=[0.7, 0.2, 0.1], shuffle=True, include_unet_outputs=False):
        """
        Fetches the data and splits into meta train, test and val
        Args:
          crop_factor: a value which says how much to crop by
          num_exposures: the number of exposures to generate when reading in images
          split: a 3 element array which says the train, val, test split -- elements must add to 1
          shuffle: a boolean variable which tells whether we should shuffle the data
          include_unet_outputs: whether to use UNET outputs as tr labels
        """
        # Number of exposures and bool flag
        self.num_exposures = num_exposures
        self.include_unet_outputs = include_unet_outputs
        
        # Fetch the data
        data = get_data(crop_factor=crop_factor, num_exposures=num_exposures, include_unet_outputs=include_unet_outputs)

        num_datapoints = data.shape[0]
        num_train = int(split[0] * num_datapoints)
        num_val = int(split[1] * num_datapoints)
        
        # Seed both
        random.seed(123)
        np.random.seed(123)
        
        indeces = np.arange(0, num_datapoints)
        if shuffle:
            random.shuffle(indeces)
        train_indeces = indeces[0:num_train]
        val_indeces = indeces[num_train:num_train+num_val]
        test_indeces = np.sort(indeces[num_train+num_val:])

        print("SPLITTING DATA INTO TRAIN, VAL, TEST...")
        self.meta_train_data = data[train_indeces]
        self.meta_test_data = data[test_indeces]
        self.meta_val_data = data[val_indeces]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
          batch_size: batch_size of data
        """
        
        data = None
        if batch_type == "meta_train":
            data = self.meta_train_data
        elif batch_type == "meta_test":
            data = self.meta_test_data
        elif batch_type == "meta_val":
            data = self.meta_val_data
        
        indeces = np.arange(0, data.shape[0])
        random_samples = np.random.choice(indeces, batch_size, replace=False)
        cur_batch = data[random_samples]
        tr_images, ts_images = [], []
        tr_labels, ts_labels = [], []
        for image_set in cur_batch:
            # Train and Test for each set of exposures
            tr, ts = train_test_split(np.arange(1, self.num_exposures+1), test_size=1)
            
            cur_tr_images, cur_tr_labels = [], []
            for i in tr:
                cur_tr_images.append(image_set[i, ...])
                cur_index = i + 3 if self.include_unet_outputs else 0
                cur_tr_labels.append(image_set[cur_index, ...])
            tr_images.append(np.stack(cur_tr_images))
            tr_labels.append(np.stack(cur_tr_labels))
            
            cur_ts_images, cur_ts_labels = [], []
            for i in ts:
                cur_ts_images.append(image_set[i, ...])
                cur_ts_labels.append(image_set[0, ...])
            ts_images.append(np.stack(cur_ts_images))
            ts_labels.append(np.stack(cur_ts_labels))
        
        tr_images = np.stack(tr_images)
        tr_labels = np.stack(tr_labels)
        ts_images = np.stack(ts_images)
        ts_labels = np.stack(ts_labels)
        
        train = np.stack([tr_images, tr_labels])
        test = np.stack([ts_images, ts_labels])
        
        return train, test