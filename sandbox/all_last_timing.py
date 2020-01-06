from thewalrus.samples import all_last_generate_hafnian_sample, all_early_generate_hafnian_sample
from thewalrus.symplectic import interferometer
import numpy as np
from scipy.linalg import qr
import sys
import time

def random_interferometer(N, real=False):
    r"""Random unitary matrix representing an interferometer.

    For more details, see arXiv:math-ph/0609050

    Args:
        N (int): number of modes
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    if real:
        z = np.random.randn(N, N)
    else:
        z = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2.0)
    q, r = qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    U = np.multiply(q, ph, q)
    return U


N = int(sys.argv[1])
num_clicks = int(sys.argv[2])

Os = interferometer(random_interferometer(N))
rs = np.random.rand(N)
cov = Os @ np.diag(np.concatenate([np.exp(-2*rs), np.exp(2*rs)])) @ Os.T


tic = time.time()
all_last_generate_hafnian_sample(cov, num_clicks)
toc = time.time()
print(N, num_clicks, toc-tic)