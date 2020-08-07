from grm import preprocessing
import eval.baselines.evaluation as evaluation
import os

predictor = "RF"
log_file = "bpi2018_application_log.csv"
name_of_case_id = "case"
name_of_activity = "event"
name_of_timestamp = "completeTime"
name_of_label = "rejected"

log = preprocessing.import_data(os.path.normpath("../../../data"), log_file, separator=",", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

evaluation.run_cross_validation(log, predictor, name_of_activity, k=10, ml_flow_run_name_prefix=log_file)
