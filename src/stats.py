import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, FP, TN, FN


def acc(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def sen(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) != 0 else 0.0


def spe(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    return TN / (TN + FP) if (TN + FP) != 0 else 0.0
