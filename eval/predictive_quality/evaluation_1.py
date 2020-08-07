"""
Evaluation No. 1

- Data Set
    - Title: BPI2017 - Loan Application Process
    - Target Variable: Loan accepted
    - Source: URL/origin
- Pre-processing
    - Filtering: -
    - Used Columns: -
"""


from eval.evaluation import run_experiment
from grm import preprocessing
from pm4py.algo.filtering.log.attributes import attributes_filter


log_file = "bpi2017.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "Accepted"
hyper_params = {'num_epochs': 1000}
k = 10

log = preprocessing.import_data("../data", log_file, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

activities = attributes_filter.get_attribute_values(log, "concept:name")

# filter out activities representing work items
w_activities = [i for i in activities.keys() if i.startswith('W_')]
log_filtered = attributes_filter.apply_events(log, w_activities, parameters={
    attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "concept:name", "positive": True})

run_experiment(log, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=log_file)
