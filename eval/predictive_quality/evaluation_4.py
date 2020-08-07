"""
Evaluation No. 4

- Data Set
    - Title: BPI2018 - Application Log
    - Target Variable: Overspend
    - Source: URL/origin
- Pre-processing
    - Filtering: -
    - Used Columns: -
"""

from eval.evaluation import run_experiment
from grm import preprocessing


logfile = "bpi2018_application_log.csv"
name_of_case_id = "case"
name_of_activity = "event"
name_of_timestamp = "completeTime"
name_of_label = "rejected"
hyper_params = {'num_epochs': 1000}
k = 10

log = preprocessing.import_data("../data", logfile, separator=",", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

run_experiment(log, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=logfile)
