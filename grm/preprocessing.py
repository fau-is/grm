import os
import random
from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.util import sorting
from pm4py.util import constants
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from grm.util import log_to_numeric, eq_dist_numeric_log, __getmapping__


def import_data(directory, file_name, separator=";", quote=None, case_id="concept:name", activity="activity",
                time_stamp="time:timestamp", target="label", num_cases=None):
    """
    Loads data from a file and returns an XLog/pm4py log object.
    Expects xes file with standard attributes and the target variable named "event: Label".
    Expects csv file with attributes "case_id", "activity", "timestamp" and "label".
    :param directory: name of path [str].
    :param file_name: name of file [str].
    :param separator: separator for csv file [char].
    :param quote: boolean flag [bool].
    :param case_id: identifier for cases [str].
    :param activity: identifier for activities [str].
    :param time_stamp: identifier for time stamps [str].
    :param target: identifier for target [str].
    :param num_cases: boolean flag [bool].
    :return: event log [EventLog].
    """

    extension = os.path.splitext(file_name)[1]
    print(os.getcwd())
    if extension == '.csv':
        data_dir = os.path.join(directory, file_name)

        # Specify column names
        CASEID_GLUE = case_id
        ACTIVITY_KEY = activity
        TIMEST_KEY = time_stamp

        parameters = {constants.PARAMETER_CONSTANT_CASEID_KEY: CASEID_GLUE,
                      constants.PARAMETER_CONSTANT_ACTIVITY_KEY: ACTIVITY_KEY,
                      constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: TIMEST_KEY,
                      'sep': separator,
                      'quotechar': quote,
                      'timest_columns': TIMEST_KEY,
                      'low_memory': False}

        # Load pm4py event stream
        event_stream = csv_importer.import_event_stream(data_dir, parameters=parameters)

        # Transform event stream to log object
        log = conversion_factory.apply(event_stream, parameters=parameters)

        # Sort log by time_stamp
        log = sorting.sort_timestamp(log, timestamp_key=TIMEST_KEY)

        # Rename to xes standard
        for trace in log:
            for event in trace:
                event.__setitem__("caseid", event.__getitem__(case_id))
                event.__setitem__("concept:name", event.__getitem__(activity))
                event.__setitem__("time:timestamp", event.__getitem__(time_stamp))
                event.__setitem__("label", event.__getitem__(target))

    elif extension == '.xes':
        data_dir = os.path.join(directory, file_name)
        log = xes_import_factory.apply(data_dir)
        print(log)
        for trace in log:
            for event in trace:
                trace.__setitem__("label", event.__getitem__(target))
    else:
        raise TypeError('File type not supported.')

    # Filter out cases where label is not set (i.e. is nan); limits number of cases in event log if set
    # util.apply(log)
    if num_cases is not None:
        log = log[:num_cases]
    print("Event log loaded")

    return log


def preprocess(log, activities):
    """
    Pre-processes an event log.
    :param log: event log as list of traces [list].
    :param activities: unique activities [list].
    :return: mapping [dic] and numeric event log as list of traces [list].
    """

    mapping = __getmapping__(log, activities)

    # convert to numeric
    num_log = log_to_numeric(log, mapping)

    # shuffle the log for splitting in training and testing
    random.shuffle(num_log)

    # split into train and test
    train = eq_dist_numeric_log(num_log[:int((len(num_log) / 10) * 9)])
    valid = eq_dist_numeric_log(num_log[int((len(num_log) / 10)) * 9:])
    raw_data = [train, valid]

    print("Created training set with " + str(len(train))
          + " cases, validation set with " + str(len(valid)))

    return mapping, raw_data
