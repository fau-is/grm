import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for tf.keras
    """

    def __init__(self, _preprocessor, list_ids, sequences, mapping_activities, max_seq_length, name_of_activity,
                 batch_size=32, shuffle=False):
        # Initialization
        self.batch_size = batch_size
        self.preprocessor = _preprocessor
        self.list_ids = list_ids
        self.sequences = sequences
        self.mapping_activities = mapping_activities
        self.max_seq_length = max_seq_length
        self.name_of_activity = name_of_activity
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """

        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """
        Generates data containing batch_size samples
        """

        # select batch of sequences (sub list)
        sequences_batch = [self.sequences[_id] for _id in list_ids_temp]

        # create tensors
        data_tensor_batch = self.preprocessor.data_tensor_from_sequences(sequences_batch, self.mapping_activities,
                                                                         self.max_seq_length, self.name_of_activity)
        label_tensor_batch = self.preprocessor.label_tensor_from_sequences(sequences_batch)

        return data_tensor_batch, label_tensor_batch
