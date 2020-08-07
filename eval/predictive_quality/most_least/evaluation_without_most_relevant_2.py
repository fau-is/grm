from eval.evaluation import run_experiment
from grm import preprocessing, GRM
from grm.util import get_activities
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.log import EventLog
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.exporter.csv import factory as csv_exporter
import operator

logfile = "sp2020.csv"
name_of_case_id = "CASE_ID"
name_of_activity = "ACTIVITY"
name_of_timestamp = "TIMESTAMP"
name_of_label = "REPAIR_IN_TIME_5D"
hyper_params = {'num_epochs': 1000}
k = 10

log = preprocessing.import_data("../data", logfile, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

# filter out most relevant activity
model_path = '../best_models/sp2020/2020-05-05-14-59_best_model.pickle'
activities = get_activities(log)
grm_model = GRM.GRM(log, activities, restore_file=model_path)

filtered_log = EventLog()
for trace in log:
    case_id, pred, rel_scores = grm_model.predict(trace)
    if len(rel_scores) > 1:
        most_relevant = max(rel_scores.items(), key=operator.itemgetter(1))[0]
        log_trace = attributes_filter.apply_events(log, [case_id], parameters={
            attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_case_id, "positive": True})

        trace_without_most = attributes_filter.apply_events(log_trace, [most_relevant], parameters={
            attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "concept:name", "positive": False})

        trace_without_most = trace_without_most[0]

        filtered_log._list.append(trace_without_most)

log = conversion_factory.apply(filtered_log)
csv_exporter.export(log, "sp2020_without_most_relevant.csv")

run_experiment(log, hyper_params=hyper_params, k=k, ml_flow_run_name_prefix=logfile)
