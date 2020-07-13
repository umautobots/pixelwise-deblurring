#!/usr/bin/env python3.6
# --------------------------------------------------------
# Pixel-Wise Deblurring
# Copyright (c) 2020 Manikandasriram S.R.
# FCAV University of Michigan
# --------------------------------------------------------
import h5py
import time
import tqdm
import argparse
import functools

import numpy as np
from multiprocessing import Pool

from src.utils import construct_problem_matrices, construct_Y
from src.optimizers import optim_wrapper, perform_lasso


def parse_args():
    parser = argparse.ArgumentParser(description='Regress from T_m to T_m^ss')
    parser.add_argument('--tau', help='thermal time constant of camera', default=0.011, type=float)
    parser.add_argument('--n', help='max level of haar wavelets', default=8, type=int)
    parser.add_argument('--N', help='number of frames used for deblurring', default=33, type=int)
    parser.add_argument('--T', help='sampling period of camera', default=0.005, type=float)
    parser.add_argument('--lmd', help='weight for sparsity', default=0.1, type=float)
    parser.add_argument('--kappa', help='weight for observations', default=0.1, type=float)
    parser.add_argument('--matfile', help='path to matfile containing image data', required=True, type=str)
    parser.add_argument('--indices', help='list of indices to deblur', nargs='*', required=True, type=int)
    parser.add_argument('--output-prefix', help='filename to save the processed frames', required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with h5py.File(args.matfile, 'r') as data:
        timestamps = data['t'][0, :] - data['t'][0, 0]  # Creating a local timescale
        obs_frames = data['frames']
        num_frames = len(timestamps)

        N = args.N
        if not int(np.log2(N - 1)) == np.log2(N - 1):
            print("Warning: Window size ({}) is not of the form 2**l + 1. Unreliable results.".format(N))

        p = Pool()
        for i in args.indices:
            print("Processing Frames: [{}, {}]".format(i, i + N))
            # Note: verify enough frames are available
            if i + N > num_frames:
                print("Skipping frame {} since the matfile doesn't have the subsequent N frames".format(i))
                continue

            obs_t = timestamps[i:i + N]
            obs_frames_subset = obs_frames[i:i + N, :, :]
            img_width, img_height = obs_frames_subset.shape[1], obs_frames_subset.shape[2]
            y_reshaped = obs_frames_subset.reshape((N, -1))
            y_mean = np.mean(y_reshaped, axis=0)

            # Note: check timestamps are uniformly spaced so
            # that they align well with the timescale of D
            # TODO: verify t
            t_samples = np.linspace(obs_t[0], obs_t[0] + (N - 1) * args.T, num=N)
            time_offset_flag = False
            if np.linalg.norm(t_samples - obs_t) > args.T:
                 time_offset_flag = True
                 print("Warning: observation timestamps don't seem to be uniformly spaced. Approximation by uniform spacing will affect results")

            V, H = construct_problem_matrices(t_samples, args.tau, args.n)
            Z = V.dot(H.transpose()).toarray()

            pixels = []
            for j in range(y_reshaped.shape[1]):
                Y = construct_Y(t_samples, y_reshaped[:, j] - y_mean[j], args.tau)
                pixels.append((Y, j))
            timescale = np.linspace(t_samples[0], t_samples[-1], 2 ** args.n, endpoint=False)
            x_est = np.zeros((timescale.size, y_reshaped.shape[1]))

            tic = time.time()
            for j, results in tqdm.tqdm(p.imap_unordered(functools.partial(optim_wrapper, func=perform_lasso, Z=Z, lmd=args.lmd, kappa=args.kappa), pixels), total=len(pixels)):
                D_est_j = results[0]
                x_est_j = y_mean[j] + H.transpose().dot(D_est_j)
                x_est[:, j] = x_est_j
            toc = time.time()
            print("Total time taken: {} for {} pixels. Average time taken: {} per pixel".format(toc - tic, len(pixels), (toc-tic)/len(pixels)))
            est_frames_subset = x_est.reshape([timescale.size, img_width, img_height])

            np.savez("{}_{:06d}_{:06d}".format(args.output_prefix, i, i + N), obs_frames=obs_frames_subset,
                     est_frames=est_frames_subset, t=t_samples, timescale=timescale)