# --------------------------------------------------------
# Pixel-Wise Deblurring
# Copyright (c) 2020 Manikandasriram S.R.
# FCAV University of Michigan
# --------------------------------------------------------
import time
import numpy as np
from scipy import sparse

import cplex
from cplex.exceptions import CplexError


def perform_lasso(Y, Z, lmd, kappa):
    try:
        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.objective.set_sense(prob.objective.sense.minimize)

        # Dimension of the coefficient vector
        K = Z.shape[1]
        Q_ = 2 * kappa * Z.transpose().dot(Z)
        Q = np.concatenate([np.concatenate([Q_, -Q_]), np.concatenate([-Q_, Q_])], axis=1)
        e = np.ones(K)
        c = np.concatenate([-2 * kappa * Z.transpose().dot(Y) + lmd * e, 2 * kappa * Z.transpose().dot(Y) + lmd * e])  # objective function
        prob.variables.add(obj=c)
        Qlil = sparse.lil_matrix(Q)
        qmat = [[row, data] for row, data in zip(Qlil.rows, Qlil.data)]
        prob.objective.set_quadratic(qmat)
    except CplexError as exc:
        print(exc)
        return

    try:
        tic = time.time()
        prob.solve()
        toc = time.time()

        qp_sol = np.array(prob.solution.get_values())
        # shrink components less than 1e-3*max(qp_sol)
        # Note: qp_sol is all positive numbers. So no need to take abs
        qp_sol[qp_sol < 1e-3 * np.max(qp_sol)] = 0

        D_pos = qp_sol[:K]
        D_neg = qp_sol[K:]
        D_star = D_pos - D_neg
    except CplexError as exc:
        D_star = np.zeros(K)
        return D_star, {'compute_time': np.inf, 'Q': Q, 'c': c, 'qp_sol': None}

    return D_star, {'compute_time': toc-tic, 'Q': Q, 'c': c, 'qp_sol': qp_sol, 'status': prob.solution.status[prob.solution.get_status()]}


def optim_wrapper(Yj, func, Z, lmd, kappa):
    return Yj[1], func(Yj[0], Z, lmd, kappa)
