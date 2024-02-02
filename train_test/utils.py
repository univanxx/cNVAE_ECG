from sklearn.metrics import confusion_matrix
import numpy as np


def find_threshold(trues, logits, eps=1e-9, beta = 2):
    metric = []
    for thr_i in logits:
        preds = logits > thr_i
        tn, fp, fn, tp = confusion_matrix(trues, preds).ravel()
        g_beta = tp / (tp+fp+beta*fn + eps)
        precision, recall = tp/(tp+fp + eps), tp/(tp+fn + eps)
        f1_beta = (1+beta*beta)*(precision+recall) / (beta*beta*precision + recall + eps)
        metric.append(f1_beta / g_beta + eps)
    return logits[np.argmax(metric)]