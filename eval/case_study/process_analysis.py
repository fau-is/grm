from grm import preprocessing, GRM
from grm.util import get_activities
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.util import constants
from pm4py.objects.log.util import sampling

model_path = '../best_models/sp2020/2020-05-06-05-40_best_model.pickle'
logfile = "sp2020.csv"
name_of_case_id = "CASE_ID"
name_of_activity = "ACTIVITY"
name_of_timestamp = "TIMESTAMP"
name_of_label = "REPAIR_IN_TIME_5D"

log = preprocessing.import_data("data", logfile, separator=";", quote='"', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

activities = get_activities(log)
grm_model = GRM.GRM(log, activities, restore_file=model_path)

log = attributes_filter.apply(log, [0],
                              parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "label", "positive": True})
log = sampling.sample(log, n=5000)
grm_model.visualize_dfg(save_file=True, log=log, file_name="sp2020_", variant="all")
