from sklearn.model_selection import KFold
import eval.baselines.training as training
import eval.baselines.testing as testing
import eval.baselines.preprocessing as preprocessing
import mlflow
import uuid
from mlflow import log_metric, log_param, log_artifact
from statistics import mean, stdev


def run_cross_validation(log, predictor, name_of_activity, hp=None,
                         k=10, seed=0, shuffle=True,
                         ml_flow_uri="databricks",
                         ml_flow_exp="/Shared/GRM-ggnn",
                         ml_flow_run_name_prefix="Experiment"):
    """
    Performs k-fold cross validation.
    :param log: name of the event log
    :param predictor:
    :param name_of_activity:
    :param hp:
    :param k:
    :param seed:
    :param shuffle:
    :param ml_flow_uri:
    :param ml_flow_exp:
    :param ml_flow_run_name_prefix:
    :return:
    """

    mlflow.set_tracking_uri(ml_flow_uri)
    mlflow.set_experiment(ml_flow_exp)

    # get meta attributes
    max_seq_length = preprocessing.max_sequence_length_from_log(log)
    mapping_activities = preprocessing.mapping_activities_from_log(log, name_of_activity)

    with mlflow.start_run(run_name=ml_flow_run_name_prefix + "_" + str(uuid.uuid1())) as run:
        if hp:
            for key, value in hp.items():
                log_param(key, value)
        log_param("k", k)

        results_measures = dict()

        k_fold = KFold(n_splits=k, random_state=seed, shuffle=shuffle)
        fold = 0
        for train_indices, test_indices in k_fold.split(log):
            fold += 1

            with mlflow.start_run(nested=True, run_name="run_%d" % fold) as run_cv:
                print("Starting Run " + str(fold))

                training.train(log, train_indices, mapping_activities, max_seq_length, predictor, fold, name_of_activity)
                measures = testing.test(log, test_indices, mapping_activities, max_seq_length, predictor, fold, name_of_activity)

                for key in measures.keys():
                    log_metric(key, measures[key], fold)
                    if key in results_measures:
                        pass
                    else:
                        results_measures[key] = []
                    results_measures[key].append(measures[key])
                    print(key + " of run " + str(fold) + ": " + str(round(measures[key], 3)))

                if predictor == "BiLSTM":
                    log_artifact('../../model/%s_%s.h5' % (predictor, fold))
                else:
                    log_artifact('../../model/%s_%s.joblib' % (predictor, fold))

                log_artifact('../../../results/cm.pdf')
                log_artifact('../../../results/gt_pred.csv')

        for key in results_measures.keys():
            overall_measure = mean(results_measures[key])
            log_metric(key, overall_measure)
            print("Overall " + key + ": " + str(overall_measure))

        overall_stdev = stdev(results_measures["accuracy"])
        log_metric("stdev", overall_stdev)
        print("Standard Deviation: " + str(overall_stdev))

