# Parts of this code were either copied or influenced by the AntroPy library
# https://github.com/raphaelvallat/antropy
# The library is under the BSD-3 License

import numpy as np


def katz_fd(x, axis=-1):
    x = np.asarray(x)
    dists = np.abs(np.diff(x, axis=axis))
    L = np.sum(dists, axis=axis)

    a = np.mean(dists, axis=axis)

    d = np.max(np.abs(x - x.take([0], axis=axis)), axis=axis)
    kfd_val = np.log10(L / a) / np.log10(d / a)
    return kfd_val
