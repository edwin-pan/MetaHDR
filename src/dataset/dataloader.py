import random
import numpy as np
from sklearn.model_selection import train_test_split

class DataGenerator(object):
    """
    Data Generator capable of generating batches of data.
    """

    def __init__(self, data, split=[0.7, 0.2, 0.1], shuffle=True):
        """
        Args:
          data: the numpy array of data which will be samples
          split: a 3 element array which says the train, test, val split -- elements must add to 1
          shuffle: a boolean variable which tells whether we should shuffle the data
        """
        num_datapoints = data.shape[0]
        num_train = int(split[0] * num_datapoints)
        num_test = int(split[1] * num_datapoints)
        
        random.seed(123)
        indeces = np.arange(0, num_datapoints)
        if shuffle:
            random.shuffle(indeces)
        train_indeces = indeces[0:num_train]
        test_indeces = indeces[num_train:num_train+num_test]
        val_indeces = indeces[num_train+num_test:]

        print("SPLITTING DATA INTO TRAIN, TEST, VAL...")
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
            # Train and Test for each set of 3 exposures
            tr, ts = train_test_split(np.array([1, 2, 3]), test_size=0.33)
            for i in tr:
                tr_images.append(image_set[i, ...])
                tr_labels.append(image_set[0, ...])
            
            for i in ts:
                ts_images.append(image_set[i, ...])
                ts_labels.append(image_set[0, ...])
        
        tr_images = np.stack(tr_images)
        tr_labels = np.stack(tr_labels)
        ts_images = np.stack(ts_images)
        ts_labels = np.stack(ts_labels)
        
        
        train = np.stack([tr_images, tr_labels])
        test = np.stack([ts_images, ts_labels])
        
        return train, test