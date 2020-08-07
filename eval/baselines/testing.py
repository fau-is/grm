import eval.baselines.preprocessing as preprocessing
import eval.util.metrics as metrics
import csv
from joblib import load
import xgboost as xgb
import tensorflow as tf


def class_from_prediction(prediction):
    """
    Get the class with highest probability from prediction.
    :param prediction:
    :return: class index
    """

    max_prob = prediction[0]

    if max_prob > prediction[1]:
        return 0
    else:
        return 1


def test(log, indices, mapping_activities, max_seq_length, predictor, fold, name_of_activity):
    """
    Perform eval.
    :param name_of_activity:
    :param log:
    :param indices:
    :param mapping_activities:
    :param max_seq_length:
    :param fold:
    :return:
    """

    sequences = preprocessing.sequences_from_log_and_indices(log, indices)
    data_tensor = preprocessing.data_tensor_from_sequences(sequences, mapping_activities, max_seq_length, name_of_activity)
    label_tensor = preprocessing.label_tensor_from_sequences(sequences)

    if predictor == "BiLSTM":
        model = tf.keras.models.load_model('../../model/%s_%s.h5' % (predictor, fold))
        predictions = model.predict(data_tensor)
        predictions = [class_from_prediction(prediction) for prediction in predictions]
    elif predictor == "RF":
        data_tensor = data_tensor.reshape(-1, max_seq_length * len(mapping_activities.keys()))
        model = load('../../model/%s_%s.joblib' % (predictor, fold))
        predictions = model.predict_proba(data_tensor)
        predictions = [class_from_prediction(prediction) for prediction in predictions]
    elif predictor == "XG":
        data_tensor = xgb.DMatrix(data_tensor.reshape(-1, max_seq_length * len(mapping_activities.keys())))
        model = load('../../model/%s_%s.joblib' % (predictor, fold))
        predictions = metrics.label_binarizer(model.predict(data_tensor))

    labels = preprocessing.label_sequence_from_tensor(label_tensor)

    # save labels and predictions
    with open('../../../results/gt_pred.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Label', 'Prediction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, prediction in zip(labels, predictions):
            writer.writerow({'Label': label, 'Prediction': prediction})

    return metrics.metrics_from_prediction_and_label(labels, predictions)








