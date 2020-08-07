from grm import preprocessing
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter


# bpi2017w____________________________________________________________________________________________________________
log_file = "bpi2017.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "Accepted"

log = preprocessing.import_data("../data", log_file, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

activities_2017 = attributes_filter.get_attribute_values(log, "concept:name")

# filter out activities representing work items
w_activities_2017 = [i for i in activities_2017.keys() if i.startswith('W_')]
fil_log_17 = attributes_filter.apply_events(log, w_activities_2017,
                                            parameters={attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY:
                                                            "concept:name", "positive": True})

# instances
print("2017 instances: ", len(fil_log_17))

# variants
variants = variants_filter.get_variants(fil_log_17)
print("2017 variants: ", len(variants))

# instances per variant
sum_val = 0
for value in variants.values():
    sum_val += len(value)
ipv = sum_val / len(variants)
print("2017 instances per variant", ipv)

# events
events_2017 = 0
for trace in fil_log_17:
    events_2017 += len(trace)
print("2017 events", events_2017)

# activities
activities = attributes_filter.get_attribute_values(fil_log_17, "concept:name")
print("2017 activities", len(activities))

# class distribution
labels = attributes_filter.get_attribute_values(fil_log_17, "Accepted")
print(labels)

trace_filter_log_pos = attributes_filter.apply(fil_log_17, [True],
                                               parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": True})
tracefilter_log_neg = attributes_filter.apply(fil_log_17, [True],
                                              parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": False})
pos = len(trace_filter_log_pos)
neg = len(tracefilter_log_neg)
print("2017 pos", pos, ", part: ", pos / (pos + neg))
print("2017 neg", neg, ", part: ", neg / (pos + neg))


# sp2020____________________________________________________________________________________________________________
log_file = "coffeemachine_service_repair.csv"
name_of_case_id = "CASE_ID"
name_of_activity = "ACTIVITY"
name_of_timestamp = "TIMESTAMP"
name_of_label = "REPAIR_IN_TIME_5D"

sp2020 = preprocessing.import_data("../data", log_file, separator=",", quote='"', case_id=name_of_case_id,
                                   activity=name_of_activity,
                                   time_stamp=name_of_timestamp, target=name_of_label)

# instances
print("2020 instances: ", len(sp2020))

# variants
variants = variants_filter.get_variants(sp2020)
print("2020 variants: ", len(variants))

# instances per variant
sum_val = 0
for value in variants.values():
    sum_val += len(value)
ipv = sum_val / len(variants)
print("2020 instances per variant", ipv)

# events
events_2020 = 0
for trace in sp2020:
    events_2020 += len(trace)
print("2020 events", events_2020)

# activities
activities = attributes_filter.get_attribute_values(sp2020, "concept:name")
print("2020 activities", len(activities))

# class distribution
labels = attributes_filter.get_attribute_values(sp2020, "Accepted")
print(labels)

trace_filter_log_pos = attributes_filter.apply(sp2020, [True],
                                               parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": True})
tracefilter_log_neg = attributes_filter.apply(sp2020, [True],
                                              parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": False})
pos = len(trace_filter_log_pos)
neg = len(tracefilter_log_neg)
print("2020 pos", pos, ", part: ", pos / (pos + neg))
print("2020 neg", neg, ", part: ", neg / (pos + neg))


# bpi2020____________________________________________________________________________________________________________
log_file = "BPI2020_PermitLog.csv"
name_of_case_id = "Case ID"
name_of_activity = "Activity"
name_of_timestamp = "Complete Timestamp"
name_of_label = "(case) Overspent"

log = preprocessing.import_data("../data", log_file, separator=";", quote='', case_id=name_of_case_id,
                                activity=name_of_activity,
                                time_stamp=name_of_timestamp, target=name_of_label)

# filter out activities representing work items
fil_2020 = attributes_filter.apply_events(log, ['STAFF MEMBER'],
                                          parameters={attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "Resource",
                                                      "positive": True})

# instances
print("2020bpi instances: ", len(fil_2020))

# variants
variants = variants_filter.get_variants(fil_2020)
print("2020bpi variants: ", len(variants))

# instances per variant
sum_val = 0
for value in variants.values():
    sum_val += len(value)
ipv = sum_val / len(variants)
print("2020bpi instances per variant", ipv)

# events
events_2020 = 0
for trace in fil_2020:
    events_2020 += len(trace)
print("2020bpi events", events_2020)

# activities
activities = attributes_filter.get_attribute_values(fil_2020, "concept:name")
print("2020bpi activities", len(activities))

# class distribution
labels = attributes_filter.get_attribute_values(fil_2020, "Accepted")
print(labels)

trace_filter_log_pos = attributes_filter.apply(fil_2020, [True],
                                               parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": True})
tracefilter_log_neg = attributes_filter.apply(fil_2020, [True],
                                              parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": False})
pos = len(trace_filter_log_pos)
neg = len(tracefilter_log_neg)
print("2020bpi pos", pos, ", part: ", pos / (pos + neg))
print("2020bpi neg", neg, ", part: ", neg / (pos + neg))


# bpi2018____________________________________________________________________________________________________________
log_file = "bpi2018_application_log.csv"
name_of_case_id = "case"
name_of_activity = "event"
name_of_timestamp = "completeTime"
name_of_label = "rejected"

log2018 = preprocessing.import_data("../data", log_file, separator=",", quote='"', case_id=name_of_case_id,
                                    activity=name_of_activity,
                                    time_stamp=name_of_timestamp, target=name_of_label)

# instances
print("2018 instances: ", len(log2018))

# variants
variants = variants_filter.get_variants(log2018)
print("2018 variants: ", len(variants))

# instances per variant
sum_val = 0
for value in variants.values():
    sum_val += len(value)
ipv = sum_val / len(variants)
print("2018 instances per variant", ipv)

# events
events_2018 = 0
for trace in log2018:
    events_2018 += len(trace)
print("2018 events", events_2018)

# activities
activities = attributes_filter.get_attribute_values(log2018, "concept:name")
print("2018 activities", len(activities))

# class distribution
labels = attributes_filter.get_attribute_values(log2018, "Accepted")
print(labels)

trace_filter_log_pos = attributes_filter.apply(log2018, [True],
                                               parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": True})
tracefilter_log_neg = attributes_filter.apply(log2018, [True],
                                              parameters={
                                                  attributes_filter.PARAMETER_CONSTANT_ATTRIBUTE_KEY: name_of_label,
                                                  "positive": False})
pos = len(trace_filter_log_pos)
neg = len(tracefilter_log_neg)
print("2018 pos", pos, ", part: ", pos / (pos + neg))
print("2018 neg", neg, ", part: ", neg / (pos + neg))
