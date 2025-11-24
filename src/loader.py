import numpy as np


def load_A_filtered(path="./A_filtered.npz"):
    data = np.load(path)
    return data[data.files[0]]


def load_E_filtered(path="./E_filtered.npz"):
    data = np.load(path)
    return data[data.files[0]]
