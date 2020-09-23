"""
Evaluation No. 5
"""

from eval.evaluation import run_experiment
from grm import preprocessing


logfile = "clickstream_anon.csv"
name_of_case_id = "CASE_KEY"
name_of_activity = "ACTIVITY"
name_of_timestamp = "EVENTTIMESTAMP"
name_of_label = "EXCEPTION"
hyper_params = {'num_epochs': 5,
                'batch_size': 1024,
                'hidden_size': 2000}
k = 10

log = preprocessing.import_data("../data", logfile, separator=",", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

run_experiment(log, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=logfile, save_artifact=False)