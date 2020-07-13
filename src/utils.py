# --------------------------------------------------------
# Pixel-Wise Deblurring
# Copyright (c) 2020 Manikandasriram S.R.
# FCAV University of Michigan
# --------------------------------------------------------
import numpy as np
from scipy import sparse


def _construct_V(t_obs, tau, K):
    N = t_obs.shape[0]
    total_time = t_obs[-1] - t_obs[0]

    M = N-1
    m = 0
    row = []
    col = []
    data = []
    for i in range(1, N):
        # add y(t_i), y(t_0) constraint
        ti_t0 = t_obs[i] - t_obs[0]
        outer_term_exponent = ti_t0 / tau
        for k in range(K):
            lower_limit = np.maximum(k*total_time/(tau*K), 0)
            upper_limit = np.minimum((k+1)*total_time/(tau*K), ti_t0/tau)
            if lower_limit < upper_limit:
                val = np.exp(upper_limit-outer_term_exponent) - np.exp(lower_limit-outer_term_exponent)
                row.append(m)
                col.append(k)
                data.append(val)
        m += 1
    assert m == M, "Number of constraints seems to be wrongly calculated"
    V = sparse.coo_matrix((data, (row, col)), shape=(M, K), dtype=np.float64)
    return V.tocsr()


def _construct_H(haar_level):
    # Size of H matrix
    K = np.power(2, haar_level)

    row = []
    col = []
    data = []

    # scaling function phi_00
    for k in range(K):
        row.append(0)
        col.append(k)
        data.append(1)

    # haar wavelets
    for l in range(1, K):
        n = np.floor(np.log2(l))
        k = l - 2**n
        # this row corresponds to \psi_{n,k} which has support in k/2**n, (k+1)/2**n with half being positive and
        # other half being negative
        # NOTE: int is used instead of floor since the indices are all positive numbers
        idx_min = int(K*k/2**n)
        idx_mid = int(K*(k+0.5)/2**n)
        idx_max = int(K*(k+1)/2**n)
        for m in range(idx_min, idx_mid):
            row.append(l)
            col.append(m)
            data.append(2**(n/2))
        for m in range(idx_mid, idx_max):
            row.append(l)
            col.append(m)
            data.append(-2**(n/2))
    H = sparse.coo_matrix((data, (row, col)), shape=(K, K), dtype=np.float64) / np.sqrt(2**haar_level)
    return H.tocsr()


def construct_Y(t_obs, y_obs, tau):
    #  y_obs are measurements in Kelvin

    assert len(y_obs.shape) == 1, "Each pixel should be independently solved"
    N = y_obs.shape[0]

    Y = []
    for i in range(1, N):
        # add y(t_i), y(t_0) constraint
        Y.append(y_obs[i] - y_obs[0] * np.exp(-(t_obs[i] - t_obs[0]) / tau))
    return np.array(Y)


def construct_problem_matrices(t_obs, tau, haar_level):
    # Util function to construct V and H given tau, haar_level and [t_0, t_1, ... , t_N]
    # t_obs are timestamps of observations in seconds
    # tau is the thermal time constant of the microbolometer in seconds
    # haar_level determines time resolution of the estimate i.e [t_0, t_N) is split into 2**haar_level bins
    # returns: V, H

    assert len(t_obs.shape) == 1 and t_obs.shape[0] >= 2, "At least two observations are required"

    V = _construct_V(t_obs, tau, 2**haar_level)
    H = _construct_H(haar_level)

    return V, H