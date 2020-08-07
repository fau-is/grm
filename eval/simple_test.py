from grm import GRM, preprocessing
from pm4py.objects.log.util import sampling

# import data
data_raw = preprocessing.import_data("data", "BPI2020_PermitLog.csv", separator=";", quote='', case_id="Case ID",
                                     activity="Activity",
                                     time_stamp="Complete Timestamp", target="(case) Overspent")

# Create new GRM model object
hyper_params = {'num_epochs': 1}
grm_model = GRM.GRM(data_raw, params=hyper_params)

# Train GGNN model
grm_model.train()

# Evaluation of the GGNN model
evaluation_metrics = grm_model.testing_log()
print(evaluation_metrics)

# Visualization as DFG (with sample of evaluation data)
multi_instance_log = sampling.sample(data_raw, n=100)
grm_model.visualize_dfg(save_file=False, log=multi_instance_log, file_name="multi")
