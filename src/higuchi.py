# Some parts of this code were influenced or copied from the AntroPy library
# https://github.com/raphaelvallat/antropy
# The library is under the BSD-3 license

import numpy as np
from math import floor

from consts import KMAX


def higuchi_fd(x, kmax=KMAX):
    x = np.asarray(x)

    n_times = x.size
    lk = np.empty(kmax)
    log_k = np.empty(kmax)

    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = int(floor((n_times - m - 1) / k))

            if n_max <= 1:
                lm[m] = 0.0
                continue

            j0 = m + np.arange(0, n_max - 1) * k
            j1 = m + np.arange(1, n_max) * k

            diffs = np.abs(x[j1] - x[j0])

            ll = diffs.sum()
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll

        lk[k - 1] = np.mean(lm)
        log_k[k - 1] = np.log(1.0 / k)

    slope, _ = np.polyfit(log_k, np.log(lk), 1)
    return slope
