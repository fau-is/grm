"""
Filter & Visualize bpi2018al
- Data Set
    - Title: BPI2018 - Application Log
    - Target Variable: Overspend
    - Source: URL/origin
"""

from grm import preprocessing, GRM
from grm.util import get_activities

log_file = "bpi2018_application_log.csv"
name_of_case_id = "case"
name_of_activity = "event"
name_of_timestamp = "completeTime"
name_of_label = "rejected"
hyper_params = {'num_epochs': 1000}
k = 10

# Load data
log = preprocessing.import_data("../data", log_file, separator=",", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

# restore trained GGNN model
grm_model = GRM.GRM(log, get_activities(log), restore_file="../predictive_quality/logged_models/2020-09-16-11-22_best_model.pickle", params=hyper_params)

# create process model (full)
grm_model.visualize_dfg(save_file=True, log=log, file_name="bpi2018_all_", variant="all")
# create process model with filter (top 5 most relevant)
grm_model.visualize_dfg(save_file=True, log=log, file_name="bpi2018_5_", variant="all", topK=5)
# create process model with filter (top 10 most relevant)
grm_model.visualize_dfg(save_file=True, log=log, file_name="bpi2018_10_", variant="all", topK=10)
