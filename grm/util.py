import numpy as np
import queue
import threading
import tensorflow as tf
from pm4py.objects.log.log import EventLog, Trace
import itertools
import random
import operator
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.util import constants
import sklearn
import pandas
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as pyplot
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory


SMALL_NUMBER = 1e-7


def multi_class_roc_auc_score(label, predict, average='weighted'):
    """
    Calculates roc auc score.
    :param label: list of labels [list]
    :param predict: list of predictions [list]
    :param average: type of calculate the average [str]
    :return: roc_auc_score [float]
    """

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(label)
    label = label_binarizer.transform(label)
    predict = label_binarizer.transform(predict)

    return sklearn.metrics.roc_auc_score(label, predict, average=average)


def multi_class_prc_auc_score(label, predict, average='weighted'):
    """
    Calculates prc auc score.
    :param label: list of labels [list]
    :param predict: list of predictions [list]
    :param average: type of calculate the average [str]
    :return: roc_auc_score [float]
    """

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(label)
    label = label_binarizer.transform(label)
    predict = label_binarizer.transform(predict)

    return sklearn.metrics.average_precision_score(label, predict, average=average)


def save_confusion_matrix(label, predict, path="cm.pdf"):
    """
    Creates confusion matrix.
    :param label: list of labels [list]
    :param predict: list of predictions [list]
    :param path: path to file [str]
    :return: none.
    """

    classes = sklearn.utils.multiclass.unique_labels(label, predict)
    cms = []
    cm = sklearn.metrics.confusion_matrix(label, predict)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        return str(n)

    cm = reduce(lambda x, y: x.add(y, fill_value=0), cms)
    annot = cm.applymap(prettify)
    cm = (cm.T / cm.sum(axis=1)).T
    fig, g = pyplot.subplots(figsize=(7, 4.5))
    g = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, rasterized=True, linewidths=0.1)
    _ = g.set(ylabel='Actual', xlabel='Prediction')

    for _, spine in g.spines.items():
        spine.set_visible(True)

    pyplot.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    pyplot.close()


def glorot_init(shape):
    """
    Returns a uniform distribution of values.
    :param shape: number of values for that the distribution is calculated [int]
    :return: distribution [list]
    """

    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator:
    """
    An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None
    """

    def __init__(self, original_iterator, max_queue_size: int = 2):
        """
        Constructor of the class.
        :param original_iterator: iterator object [object]
        :param max_queue_size: max size of queue [int]
        """
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class MLP(object):
    """
    Class representing a multi-layer perceptron (MLP).
    """
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        """
        Inits the weights of the MLP
        :param shape: size of the weight vector [int]
        :return: vector of weights [list]
        """

        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


def get_unique_trace_attribute_values(log, trace_attribute):
    """
    Gets all string trace attribute values representations for a log
    :param log: list of traces [list]
    :param trace_attribute: attributes of a trace [object]
    :return: unique_trace_attribute_values [list]
    """

    values = set()
    for trace in log:
        values.add(get_string_trace_attribute_rep(trace, trace_attribute))
    return list(sorted(values))


def get_string_trace_attribute_rep(trace, trace_attribute):
    """
    Get a representation of the feature name associated to a string trace attribute value

    Parameters
    ------------
    trace
        Trace of the log
    trace_attribute
        Attribute of the trace to consider
    Returns
    ------------
    rep
        Representation of the feature name associated to a string trace attribute value
    """
    if trace_attribute in trace.attributes:
        return trace.attributes[trace_attribute]
    return "UNDEFINED"


def get_string_event_attribute_rep(event, event_attribute):
    """
    Get a representation of the feature name associated to a string event attribute value
    Parameters
    ------------
    event
        Single event of a trace
    event_attribute
        Event attribute to consider
    Returns
    ------------
    rep
        Representation of the feature name associated to a string event attribute value
    """
    return event[event_attribute]


def get_values_event_attribute_for_trace(trace, event_attribute):
    """
    Get all the representations for the events of a trace associated to a string event attribute values
    Parameters
    -------------
    trace
        Trace of the log
    event_attribute
        Event attribute to consider
    Returns
    -------------
    values
        All feature names present for the given attribute in the given trace
    """
    values_trace = set()
    for event in trace:
        if event_attribute in event:
            values_trace.add(get_string_event_attribute_rep(event, event_attribute))
    if not values_trace:
        values_trace.add("UNDEFINED")
    return values_trace


def get_unique_event_attribute_values(log, event_attribute):
    """
    Get all the representations for all the traces of the log associated to a string event attribute values
    Parameters
    ------------
    log
        Trace of the log
    event_attribute
        Event attribute to consider
    Returns
    ------------
    values
        All feature names present for the given attribute in the given log
    """
    values = set()
    for trace in log:
        values = values.union(get_values_event_attribute_for_trace(trace, event_attribute))
    return list(sorted(values))


def merge_dict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = list(value) + list(dict1[key])
    return dict3


def avg_dict(dict1):
    """
    Calculates the average per key for a dictionary
    :param dict1: dict where averages are caluclated
    :return: dictionary with rounded values in keys
    """
    for key, value in dict1.items():
        dict1[key] = round(sum(dict1[key]) / len(dict1[key]), 3)
    return dict1


def norm_mean_dict(dict1):
    """
    Mean normalization for whole dictonary
    :param dict1: input dictonary
    :return: mean normalized dictionary values
    """
    for key, value in dict1.items():
        dict1[key] = (dict1[key] - sum(dict1.values()) / len(dict1.values())) / (
                max(dict1.values()) - min(dict1.values()))
    return dict1


def norm_min_max_dict(dict1):
    """
    min-max normalization for dictionary
    :param dict1: input dictionary
    :return: min-max normalized dictionary
    """
    for key, value in dict1.items():
        dict1[key] = round((dict1[key] - min(dict1.values())) / (max(dict1.values()) - min(dict1.values())), 3)
    return dict1


def apply(log):
    """
    Filter log by keeping only traces where label is not nan - adapted from pm4py filtering method
    values list
    Parameters
    -----------
    log
        Trace log
    values
        Allowed attributes
    parameters
        Parameters of the algorithm, including:
            activity_key -> Attribute identifying the activity in the log
            positive -> Indicate if events should be kept/removed
    Returns
    -----------
    filtered_log
        Filtered log
    """

    attribute_key = "label"
    positive = False

    filtered_log = EventLog()
    for trace in log:
        new_trace = Trace()

        found = False
        for j in range(len(trace)):
            if attribute_key in trace[j]:
                attribute_value = trace[j][attribute_key]
                if np.isnan(attribute_value):
                    found = True

        if (found and positive) or (not found and not positive):
            new_trace = trace
        else:
            for attr in trace.attributes:
                new_trace.attributes[attr] = trace.attributes[attr]

        if len(new_trace) > 0:
            filtered_log.append(new_trace)
    return filtered_log



def log_to_numeric(log, mapping):
    """
    Utility functions for internal log handling. Encodes an array numeriaclly with a given mapping
    :param log: log to be encoded
    :param mapping: mapping of activities
    :return: numerically encoded log
    """
    num_log = []
    for trace in log:
        caseid_num = __getmappingcases__(mapping, trace.attributes["concept:name"])
        for event in trace:
            activity_num = __getmappingactivities__(mapping, event.__getitem__("concept:name"))
            label_num = __getmappinglabel__(mapping, event.__getitem__("label"))
            toadd = [caseid_num, activity_num, event.__getitem__("time:timestamp"), label_num]
            num_log.append(toadd)
    new_list = [
        [instance[1:] for instance in sorted(group, key=lambda x: x[2])]
        for group in
        [list(g) for k, g in itertools.groupby(sorted(num_log, key=operator.itemgetter(0)), operator.itemgetter(0))]]
    return new_list


def __getmapping__(log, activities):
    """
    Creates the mapping dictionary from a given event log.
    :param log: event log
    :param activities: activities in the event log
    :return: mapping dictionary
    """
    mapping = dict()
    mapping["cases"] = get_unique_trace_attribute_values(log, "concept:name")
    mapping["activities"] = activities
    mapping["labels"] = get_unique_event_attribute_values(log, "label")
    return mapping


def __getmappingcases__(mapping, value):
    """
    Gets index of case value in mapping.
    :param mapping: mapping dictionary
    :param value: value for which the mapping is searched for
    :return: index of value in mapping
    """
    if value in mapping["cases"]:
        return mapping["cases"].index(value)
    else:
        return len(mapping["cases"]) + 1


def __getmappingactivities__(mapping, value):
    """
    Gets index of activity value in mapping.
    :param mapping: mapping dictionary
    :param value: value for which the mapping is searched for
    :return: index of value in mapping
    """
    if value in mapping["activities"]:
        return mapping["activities"].index(value)
    else:
        raise KeyError('Activity "' + str(value) + '" not known.')


def __getmappinglabel__(mapping, value):
    """
    Gets index of label value in mapping.
    :param mapping: mapping dictionary
    :param value: value for which the mapping is searched for
    :return: index of value in mapping
    """
    if value in mapping["labels"]:
        return mapping["labels"].index(value)
    else:
        raise KeyError('Label "' + str(value) + '" not known.')


def eq_dist_numeric_log(new_list):
    """
    Normalizes a list to use the same amount of positive and negative labels.
    :param new_list: input list
    :return: normalized list
    """
    ones, zeroes = split_to_labels(new_list)

    eq_amount = min(len(ones), len(zeroes))
    eq_distr_data = []
    for i in range(eq_amount):
        eq_distr_data.append(ones[i])
        eq_distr_data.append(zeroes[i])
    random.shuffle(eq_distr_data)
    return eq_distr_data


def split_to_labels(list):
    """
    sorts list into labels
    :param list: input lists
    :return: two lists, one per label
    """
    ones = [part for part in list if int(max([int(tt[-1]) for tt in part])) == 1]
    zeroes = [part for part in list if int(max([int(tt[-1]) for tt in part])) == 0]
    return ones, zeroes


def split_per_label(log, mapping):
    """
    sorts log into labels
    :param log: input log
    :param mapping: log mapping
    :return: two logs, one per label
    """
    ones = EventLog()
    zeros = EventLog()
    for trace in log:
        for event in trace:
            label = event.__getitem__("label")
            if label == 1:
                ones.append(trace)
            if label == 0:
                zeros.append(trace)
            break

    return ones, zeros


def split_train_test(data_raw):
    """
    splits data into 3:7 split
    :param data_raw: raw data
    :return: return train and test data
    """
    data_train = data_raw[int(len(data_raw) / 10 * 3):]
    data_test = data_raw[:int(len(data_raw) / 10 * 3)]
    return data_train, data_test


def get_activities(data):
    """
    Filteres event log to only return attribute names.
    :param data: event log
    :return: event log activities
    """
    return list(sorted(attributes_filter.get_attribute_values(data, "concept:name").keys()))


def create_pig(num_log, num_different_activities):
    """
    Packages Process Instance Graphs for the use in the GRM.
    :param num_log: the preprocessed event log
    :param num_different_activities: the number of different activities in the unprocessed event log (due to folds the
    processed event log can contain less different activities)
    :return: set of Process Instance Graphs contained in num_log
    """
    processed_data = []
    for i, trace in enumerate(num_log):
        target = int(max([int(tt[-1]) for tt in trace]))

        # note an edge is a tuple (a,b,c) with a = current node, b = type of edge, c = target node
        edges, features = make_edges(trace, num_different_activities)
        if len(edges) < 2:
            continue
        if target < 0:
            continue
        processed_data.append({
            'targets': [[target]],
            'graph': edges,
            'node_features': features
        })
    return processed_data


def make_edges(trace, num_different_activities):
    """
    Makes edges, i.e. computes Process Instance Graphs.
    Edge types:
    - start edge -> type 1
    - end edge -> type 2
    - forward edge -> type 3
    - backward edge -> type 4
    - recursive edge -> type 5
    All edges are directed edges.

    :param trace: specific trace from an event log
    :param num_different_activities: number of different activities [int]
    :return: set of edges and the features for the given trace
    """

    edges = []
    prev = -1
    node_ids = [0]
    for nbr, instance in enumerate(trace):
        # start edge
        if nbr == 0:
            edges.append((prev + 1, 1, int(instance[0]) + 1))
        # recursive edge
        elif prev == int(instance[0]) and not edges.__contains__((int(instance[0]) + 1, 5, int(instance[0]) + 1)):
            edges.append((int(instance[0]) + 1, 5, int(instance[0]) + 1))
            if edges.__contains__((int(instance[0]) + 1, 3, int(instance[0]) + 1)):
                edges.remove((int(instance[0]) + 1, 3, int(instance[0]) + 1))
        elif edges.__contains__((int(instance[0]) + 1, 3, prev + 1)) \
                and not edges.__contains__((int(instance[0]) + 1, 4, prev + 1)) \
                and not edges.__contains__((prev + 1, 4, int(instance[0]) + 1)) \
                and not edges.__contains__((prev + 1, 3, int(instance[0]) + 1)):
            edges.append((prev + 1, 4, int(instance[0]) + 1))
            edges.append((int(instance[0]) + 1, 4, prev + 1))
            if edges.__contains__((prev + 1, 3, int(instance[0]) + 1)):
                edges.remove((prev + 1, 3, int(instance[0]) + 1))
            if edges.__contains__((int(instance[0]) + 1, 3, prev + 1)):
                edges.remove((int(instance[0]) + 1, 3, prev + 1))
        elif not edges.__contains__((prev + 1, 3, int(instance[0]) + 1)) and not edges.__contains__(
                (prev + 1, 5, int(instance[0]) + 1)) and not edges.__contains__((prev + 1, 4, int(instance[0]) + 1)):
            edges.append((prev + 1, 3, int(instance[0]) + 1))

        prev = int(instance[0])
        node_ids.append(int(instance[0]) + 1)
    edges.append((prev + 1, 2, 0))

    # giving the nodes their label as feature -> Label information gets lost, only topology is regarded. here the Event ID is preserved.
    features = []
    for k in range(max(node_ids) + 1):
        feature = []
        for k in range(num_different_activities + 1):
            feature.append(int(0))
        features.append(feature)
    for node in node_ids:
        feature = []
        for k in range(num_different_activities + 1):
            feature.append(int(0))
        feature[node] = 1
        features[node] = feature
    return edges, features


def filter_log_by_caseid(log, values):
    """
    Filters log by case ID.
    :param log: log to be filtered
    :param values: value that should be filtered
    :return: filtered log
    """
    parameters = {constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "caseid"}
    return attributes_filter.apply(log, values, parameters=parameters)


def visualize_dfg(log, filename):
    """
    Visualizes an event log as a DFG
    :param log: event log that will be visualized
    :param filename: filename for the created DFG
    """
    dfg = dfg_factory.apply(log)
    parameters = {"format": "svg"}
    gviz = dfg_vis_factory.apply(dfg, log=log, parameters=parameters, variant='frequency')
    dfg_vis_factory.save(gviz, filename)

def bool_to_bin(bool_value):
    """
    Helper function to map a boolean value to a binary value.
    :param bool_value: boolean boolean value [bool]
    :return: binary value [int]
    """
    if bool_value:
        return 1
    else:
        return 0
