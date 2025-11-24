import numpy as np


def z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev
