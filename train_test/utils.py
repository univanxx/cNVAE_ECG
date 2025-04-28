from sklearn.metrics import roc_auc_score
import numpy as np


def find_threshold(trues, logits):
    thrshs = []
    for class_i in range(trues.shape[1]):
        metric = []
        for thr_i in logits[:,class_i]:
            preds = logits[:,class_i] > thr_i
            auc = roc_auc_score(trues[:, class_i], preds)
            metric.append(auc)
        thrshs.append(logits[np.argmax(metric), class_i])
    return thrshs