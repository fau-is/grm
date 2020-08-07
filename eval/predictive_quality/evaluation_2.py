"""
Evaluation No. 2

- Data Set
    - Title: Customer Service Repair Process
    - Target Variable: Repair on Time
    - Source: URL/origin
- Pre-processing
    - Filtering: -
    - Used Columns: -
"""


from eval.evaluation import run_experiment
from grm import preprocessing


log_file = "sp2020.csv"
name_of_case_id = "CASE_ID"
name_of_activity = "ACTIVITY"
name_of_timestamp = "TIMESTAMP"
name_of_label = "REPAIR_IN_TIME_5D"
hyper_params = {'num_epochs': 1000}
k = 10

log = preprocessing.import_data("../data", log_file, separator=";", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

run_experiment(log, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=log_file)
