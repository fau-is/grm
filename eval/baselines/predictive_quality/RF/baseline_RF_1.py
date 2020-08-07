from grm import preprocessing
import eval.baselines.evaluation as evaluation
from pm4py.algo.filtering.log.attributes import attributes_filter
import os

predictor = "RF"
log_file = "bpi2017.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "Accepted"

log = preprocessing.import_data(os.path.normpath("../../../data"), log_file, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

activities = attributes_filter.get_attribute_values(log, "concept:name")

# filter out activities representing work items
w_activities = [i for i in activities.keys() if i.startswith('W_')]
log_filtered = attributes_filter.apply_events(log, w_activities,
                                              parameters={attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY:
                                                              "concept:name", "positive": True})

evaluation.run_cross_validation(log_filtered, predictor, name_of_activity, k=10, ml_flow_run_name_prefix=log_file)
