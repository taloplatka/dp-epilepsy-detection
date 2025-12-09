import numpy as np
from loader import load_D_filtered, load_E_filtered
from helpers import extract_fd_features, run_knn_with_results

D_filtered = load_D_filtered()
E_filtered = load_E_filtered()


def extract_vectors(filtered_data, normalize):
    vectors = []
    for eeg in filtered_data:
        results = extract_fd_features(
            eeg,
            want_higuchi=False,
            want_katz=True,
            want_means=False,
            want_vectors=True,
            normalize=normalize,
        )
        vectors.append(results["katz_vector"])
    return np.array(vectors)


# Run for both cases
for mode in [False, True]:
    D_vecs = extract_vectors(D_filtered, normalize=mode)
    E_vecs = extract_vectors(E_filtered, normalize=mode)

    X = np.vstack([D_vecs, E_vecs])
    y = np.array([0] * 100 + [1] * 100)

    accs, sens, spes = [], [], []
    for seed in range(25):
        acc, sen, spe = run_knn_with_results(X, y, seed=seed)
        accs.append(acc)
        sens.append(sen)
        spes.append(spe)

    label = "with normalization" if mode else "without normalization"
    print(f"\nKatz {label}:")
    print("Mean accuracy: ", np.mean(accs))
    print("Mean sensitivity:", np.mean(sens))
    print("Mean specificity:", np.mean(spes))
    print("Std accuracy:   ", np.std(accs))
    print("Std sensitivity:", np.std(sens))
    print("Std specificity:", np.std(spes))
