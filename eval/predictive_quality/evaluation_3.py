"""
Evaluation No. 3

- Data Set
    - Title: BPI2020 - Permit Log - TU/e Travel Expense Reimbursement
    - Target Variable: Overspend
    - Source: URL/origin
- Pre-processing
    - Filtering: -
    - Used Columns: -
"""

from eval.evaluation import run_experiment
from grm import preprocessing
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.util import sampling

logfile = "BPI2020_PermitLog.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "(case) Overspent"
hyper_params = {'num_epochs': 1000}
k = 10

log = preprocessing.import_data("../data", logfile, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

# log = sampling.sample(log, n=100)

# filter only activities representing work items
log_filtered = attributes_filter.apply_events(log, ['STAFF MEMBER'],
                                              parameters={attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY:
                                                              "Resource", "positive": True})

run_experiment(log_filtered, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=logfile)
