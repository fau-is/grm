from grm import preprocessing
import eval.baselines.evaluation as evaluation
import os

predictor = "RF"
log_file = "BPI2020_PermitLog.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "(case) Overspent"


log = preprocessing.import_data(os.path.normpath("../../../data"), log_file, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

evaluation.run_cross_validation(log, predictor, name_of_activity, k=10, ml_flow_run_name_prefix=log_file)
