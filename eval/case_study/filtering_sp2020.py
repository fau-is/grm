"""
Filter & Visualize sp2020

- Data Set
    - Title: Customer Service Repair Process
    - Target Variable: Repair on Time
    - Source: URL/origin
"""

from grm import preprocessing, GRM
from grm.util import get_activities

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

grm_model = GRM.GRM(log, get_activities(log), restore_file="../predictive_quality/logged_models/2020-07-31-19-00_best_model.pickle", params=hyper_params)

# Example of trace log[2] with case 3 in log
# grm_model.show_most_relevant_trace(log, log[2], 3, name_of_case_id, file_name="three")
# grm_model.show_most_relevant_trace(log, log[2], 5, name_of_case_id, file_name="five")

grm_model.visualize_dfg(save_file=True, log=log, file_name="sp2020_all_", variant="all")
# Example of all relevant activities
grm_model.show_most_relevant(log, 5, save_file=True, file_name="./sp2020/sp2020_5_", variant="all")
grm_model.show_most_relevant(log, 10, save_file=True, file_name="./sp2020/sp2020_10_", variant="all")
