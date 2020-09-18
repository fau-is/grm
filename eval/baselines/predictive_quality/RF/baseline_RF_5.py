from grm import preprocessing
import eval.baselines.evaluation as evaluation
import os


predictor = "RF"
log_file = "clickstream_anon.csv"
name_of_case_id = "CASE_KEY"
name_of_activity = "ACTIVITY"
name_of_timestamp = "EVENTTIMESTAMP"
name_of_label = "EXCEPTION"

log = preprocessing.import_data(os.path.normpath("../../../data"), log_file, separator=",", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

evaluation.run_cross_validation(log, predictor, name_of_activity, k=10, ml_flow_run_name_prefix=log_file)





