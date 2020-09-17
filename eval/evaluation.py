from grm import GRM
from grm.util import get_activities
from pm4py.objects.log.util import sampling
from statistics import mean, stdev
import uuid
from sklearn.model_selection import KFold
import mlflow
from mlflow import log_metric, log_param, log_artifact


def run_experiment(data_raw, hyper_params=None, k=10, ml_flow_uri="databricks",
                   ml_flow_exp="/Shared/grm-review",
                   ml_flow_run_name_prefix="Experiment"):
    """
    Performs experiment.
    :param data_raw: raw data from event log file.
    :param hyper_params: set of hyper-parameters.
    :param k: index of k-fold cross-validation.
    :param ml_flow_uri: ??
    :param ml_flow_exp: ??
    :param ml_flow_run_name_prefix: ??
    :return: none.
    """

    # init ml flow
    mlflow.set_tracking_uri(ml_flow_uri)
    mlflow.set_experiment(ml_flow_exp)

    # load event log
    activities = get_activities(data_raw)
    num_activities = len(activities)
    with mlflow.start_run(run_name=ml_flow_run_name_prefix + "_" + str(uuid.uuid1())) as run:
        if hyper_params:
            for key, value in hyper_params.items():
                log_param(key, value)

        log_param("k", k)
        log_metric("number of activities", num_activities)
        results_measures = dict()
        i = 0

        # Perform k-fold cross-validation
        kf = KFold(n_splits=k, shuffle=True)
        for train_idx, test_idx in kf.split(data_raw):
            i += 1
            data_training = [data_raw[j] for j in train_idx]
            data_testing = [data_raw[j] for j in test_idx]

            with mlflow.start_run(nested=True, run_name="run_%d" % i) as run_cv:
                print("Starting Run " + str(i))

                # Create new GGNN model object
                grm_model = GRM.GRM(data_training, activities, restore_file=None, params=hyper_params)

                # Train GGNN model
                grm_model.train()

                # Perform evaluation
                measures = grm_model.testing_log(data_testing)
                for key in measures.keys():
                    log_metric(key, measures[key], i)
                    if key in results_measures:
                        pass
                    else:
                        results_measures[key] = []
                    results_measures[key].append(measures[key])
                    print(key + " of run " + str(i) + ": " + str(round(measures[key], 3)))

                log_artifact(grm_model.best_model_file)
                log_artifact('../results/cm.pdf')

        for key in results_measures.keys():
            overall_measure = mean(results_measures[key])
            log_metric(key, overall_measure)
            print("Overall " + key + ": " + str(overall_measure))

        overall_st_dev = stdev(results_measures["accuracy"])
        log_metric("st_dev", overall_st_dev)
        print("Standard deviation: " + str(overall_st_dev))

        """ Relevance visualisation for one instance """
        # Extract one random instance from the log
        single_instance_log = sampling.sample(data_raw, n=1)

        # Visualization as direct follower graph (DFG) with evaluation data
        filenames = grm_model.visualize_dfg(save_file=True, log=single_instance_log, file_name="single")
        for file in filenames:
            log_artifact(file)

        """ Relevance visualisation for 1000 instances """
        # Extract 1000 instances from the event log
        multi_instance_log = sampling.sample(data_raw, n=1000)

        # Visualization as DFG (with evaluation data)
        for file in grm_model.visualize_dfg(save_file=True, log=multi_instance_log, file_name="multi"):
            log_artifact(file)
