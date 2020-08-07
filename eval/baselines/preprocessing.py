import numpy as np
import sklearn


def sequences_from_log_and_indices(log, indices):
    """
    Creates sequences from log and indices.
    :param log: event log.
    :param indices: indices of instances in the event log.
    :return: list of sequences.
    """

    sequences = []

    for index in indices:
        sequences.append(log[index])

    return sequences


def data_tensor_from_sequences(sequences, mapping_activities, max_seq_length, name_of_activity):
    """
    Creates data tensor of third order from sequences.
    :param name_of_activity: name of the activity.
    :param max_seq_length: length of the longest sequence in the event log.
    :param mapping_activities: mapping of activities to indices.
    :param sequences: list of sequences.
    :return: data_tensor.
    """

    data_tensor = np.zeros((len(sequences), max_seq_length, len(mapping_activities.keys())), dtype=np.float32)

    for index, sequence in enumerate(sequences):
        for index_, event in enumerate(sequence):
            index_activity = mapping_activities[event[name_of_activity]]
            data_tensor[index, index_, index_activity] = 1

    return data_tensor


def label_tensor_from_sequences(sequences, num_classes=2):
    """
    Creates label tensor of second order (matrix) from sequences.
    :param num_classes: number of classes.
    :param sequences: list of sequences.
    :return: label_tensor.
    """

    label_tensor = np.zeros((len(sequences), num_classes), dtype=np.int_)

    for index, sequence in enumerate(sequences):
        for index_, event in enumerate(sequence):
            if event['label'] == 0:
                label_tensor[index, 0] = 1
            else:
                label_tensor[index, 1] = 1

    return label_tensor


def max_sequence_length_from_log(log):
    """
    Returns length of the longest sequence in the event log.
    :param log: event log.
    :return: max_seq_length.
    """

    max_seq_length = 0

    for sequence in log:
        max_seq_length_temp = 0
        for activity_ in sequence:
            max_seq_length_temp += 1
        if max_seq_length_temp > max_seq_length:
            max_seq_length = max_seq_length_temp

    return max_seq_length


def unique_activities_from_log(log, name_of_activity):
    """
    Returns unique activities from event log.
    :param name_of_activity: name of activity.
    :param log: event log.
    :return: unique activities.
    """

    unique_activities = []

    for sequence in log:
        for activity in sequence:
            unique_activities.append(activity[name_of_activity])

    return sorted(list(set(unique_activities)))


def mapping_activities_from_log(log, name_of_activity):
    """
    Returns mapping activities of activities.
    :param name_of_activity:
    :param log:
    :return: mapping
    """

    mapping_activities = dict()
    unique_activities = unique_activities_from_log(log, name_of_activity)

    for index, activity in enumerate(unique_activities):
        mapping_activities[activity] = index

    return mapping_activities


def label_sequence_from_tensor(tensor):
    """
    Returns vector with label values from tensor of second order.
    :param tensor: label tensor of second order.
    :return: sequence of labels.
    """

    labels = []

    for index in range(0, tensor.shape[0]):
        if tensor[index, 0] == 1:
            labels.append(0)
        else:
            labels.append(1)

    return labels


def create_index_from_sequences(sequences):
    """
    Creates indices from sequences.
    :param sequences: list of sequences.
    :return: indices.
    """

    indices = []
    for index in range(0, len(sequences)):
        indices.append(index)
    return indices


def split_train_test(sequences):
    """
    Splits data set into train- and test set.
    :param sequences: list of sequences.
    :return: sequences for training and testing.
    """

    return sklearn.model_selection.train_test_split(sequences, sequences, test_size=0.1, random_state=0)
    

