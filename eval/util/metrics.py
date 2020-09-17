import sklearn
import pandas
import seaborn as sns
import matplotlib.pyplot as pyplot
from functools import reduce
# import numpy as np


def metrics_from_prediction_and_label(labels, predictions, verbose=False):

    measures = {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(labels, predictions),
        "precision_micro": sklearn.metrics.precision_score(labels, predictions, average='micro'),
        "precision_macro": sklearn.metrics.precision_score(labels, predictions, average='macro'),
        "precision_weighted": sklearn.metrics.precision_score(labels, predictions, average='weighted'),
        "recall_micro": sklearn.metrics.recall_score(labels, predictions, average='micro'),
        "recall_macro": sklearn.metrics.recall_score(labels, predictions, average='macro'),
        "recall_weighted": sklearn.metrics.recall_score(labels, predictions, average='weighted'),
        "f1_score_micro": sklearn.metrics.f1_score(labels, predictions, average='micro'),
        "f1_score_macro": sklearn.metrics.f1_score(labels, predictions, average='macro'),
        "f1_score_weighted": sklearn.metrics.f1_score(labels, predictions, average='weighted')
    }

    try:
        measures["roc_auc_weighted"] = multi_class_roc_auc_score(labels, predictions, 'weighted')
        measures["roc_auc_macro"] = multi_class_roc_auc_score(labels, predictions, 'macro')
        measures["roc_auc_micro"] = multi_class_roc_auc_score(labels, predictions, 'micro')
    except ValueError:
        print("Warning: Roc auc score can not be calculated ...")

    try:
        # note we use the average precision at different threshold values as the auc of the pr-curve
        # and not the auc-pr-curve with the trapezoidal rule / linear interpolation because it could be too optimistic
        measures["auc_prc_weighted"] = multi_class_prc_auc_score(labels, predictions, 'weighted')
        measures["auc_prc_macro"] = multi_class_prc_auc_score(labels, predictions, 'macro')
        measures["auc_prc_micro"] = multi_class_prc_auc_score(labels, predictions, 'micro')
    except ValueError:
        print("Warning: Auc prc score can not be calculated ...")

    save_confusion_matrix(labels, predictions)

    report = save_classification_report(labels, predictions)
    classes = list(sorted(set(labels)))
    for pos_class in classes:
        measures[str(pos_class) + "_precision"] = report[str(pos_class)]['precision']
        measures[str(pos_class) + "_recall"] = report[str(pos_class)]['recall']
        measures[str(pos_class) + "_f1-score"] = report[str(pos_class)]['f1-score']
        measures[str(pos_class) + "_support"] = report[str(pos_class)]['support']

        if pos_class == 1:
            neg_class = 0
        else:
            neg_class = 1

        tp, fp, tn, fn = calculate_cm_states(labels, predictions, pos_class, neg_class)

        measures[str(pos_class) + "_tp"] = tp
        measures[str(pos_class) + "_fp"] = fp
        measures[str(pos_class) + "_tn"] = tn
        measures[str(pos_class) + "_fn"] = fn


        if tn + fp == 0:
            pass
        else:
            # Specificity or true negative rate
            measures[str(pos_class) + "_tnr"] = tn / (tn + fp)

            # Fall out or false positive rate
            measures[str(pos_class) + "_fpr"] = fp / (fp + tn)

        if tn + fn == 0:
            pass
        else:
            # Negative predictive value
            measures[str(pos_class) + "_npv"] = tn / (tn + fn)

        if tp + fn == 0:
            pass
        else:
            # False negative rate
            measures[str(pos_class) + "_fnr"] = fn / (tp + fn)

        if tp + fp == 0:
            pass
        else:
            # False discovery rate
            measures[str(pos_class) + "_fdr"] = fp / (tp + fp)

    return measures


def calculate_cm_states(labels, predictions, pos_class, neg_class):

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predictions)):
        if labels[i] == predictions[i] == pos_class:
            tp += 1
        if predictions[i] == pos_class and labels[i] != predictions[i]:
            fp += 1
        if labels[i] == predictions[i] == neg_class:
            tn += 1
        if predictions[i] == neg_class and labels[i] != predictions[i]:
            fn += 1

    return tp, fp, tn, fn


def save_classification_report(labels, predictions):
    return sklearn.metrics.classification_report(y_true=labels, y_pred=predictions, output_dict=True)


def multi_class_roc_auc_score(label, predict, average):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(label)
    label = label_binarizer.transform(label)
    predict = label_binarizer.transform(predict)

    return sklearn.metrics.roc_auc_score(label, predict, average=average)


def multi_class_prc_auc_score(label, predict, average):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(label)
    label = label_binarizer.transform(label)
    predict = label_binarizer.transform(predict)

    return sklearn.metrics.average_precision_score(label, predict, average=average)


def label_binarizer(labels):

    for index in range(0, len(labels)):
        if labels[index] >= 0.5:
            labels[index] = 1.0
        else:
            labels[index] = 0.0

    return labels


def save_confusion_matrix(labels, predictions, path="../results/cm.pdf"):

    classes = sklearn.utils.multiclass.unique_labels(labels, predictions)
    cms = []
    cm = sklearn.metrics.confusion_matrix(labels, predictions)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        """
        if n > 1000000:
            return str(np.round(n / 1000000, 1)) + 'M'
        elif n > 1000:
            return str(np.round(n / 1000, 1)) + 'K'
        else:
            return str(n)
        """
        return str(n)

    cm = reduce(lambda x, y: x.add(y, fill_value=0), cms)
    annot = cm.applymap(prettify)
    cm = (cm.T / cm.sum(axis=1)).T
    fig, g = pyplot.subplots(figsize=(7, 4.5))
    g = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, rasterized=True, linewidths=0.1)
    _ = g.set(ylabel='Actual', xlabel='Prediction')

    for _, spine in g.spines.items():
        spine.set_visible(True)

    pyplot.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    pyplot.close()