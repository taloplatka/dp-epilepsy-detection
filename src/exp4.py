# This experiment examines how Higuchi's stats change as its hyperparameter kmax changes
# We check from kmax = 5 to kmax = 50 in jumps of 5

import numpy as np

from loader import load_D_filtered, load_E_filtered
from helpers import run_knn_with_results, extract_fd_features
from consts import K

D_filtered = load_D_filtered()
E_filtered = load_E_filtered()


def calculate_with_kmax(kmax):
    D_hfd = []
    E_hfd = []

    for eeg in D_filtered:
        results = extract_fd_features(
            eeg,
            want_higuchi=True,
            want_katz=False,
            want_means=False,
            want_vectors=True,
            kmax=kmax,
        )
        D_hfd.append(results["higuchi_vector"])

    for eeg in E_filtered:
        results = extract_fd_features(
            eeg,
            want_higuchi=True,
            want_katz=False,
            want_means=False,
            want_vectors=True,
            kmax=kmax,
        )
        E_hfd.append(results["higuchi_vector"])

    D_hfd = np.array(D_hfd)
    E_hfd = np.array(E_hfd)
    return D_hfd, E_hfd


for i in range(5, 55, 5):
    D_hfd, E_hfd = calculate_with_kmax(i)
    X = np.vstack([D_hfd, E_hfd])
    y = np.array([0] * 100 + [1] * 100)

    accs, sens, spes = [], [], []
    for seed in range(25):
        acc, sen, spe = run_knn_with_results(X, y, seed=seed)
        accs.append(acc)
        sens.append(sen)
        spes.append(spe)

    print(
        f"Higuchi with kmax = {i} yields mean accuracy of {np.mean(accs)}, mean sensitivity of {np.mean(sens)} and mean specificity of {np.mean(spes)}\n"
    )
