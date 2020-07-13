#!/usr/bin/env python3.6
# --------------------------------------------------------
# Pixel-Wise Deblurring
# Copyright (c) 2020 Manikandasriram S.R.
# FCAV University of Michigan
# --------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist, equalize_adapthist


parser = argparse.ArgumentParser(description='View files saved by deblurring algorithms for viz')
parser.add_argument('--equalize', help='whether to apply CLAHE', default=1, type=int)
parser.add_argument('--filename', help='path to npz file', type=str)
parser.add_argument('--clip_min', help='clips the frames before normalizing and equalizing', type=float)
parser.add_argument('--clip_max', help='clips the frames before normalizing and equalizing', type=float)
args = parser.parse_args()

with np.load(args.filename) as data:
    obs_frames = data['obs_frames']
    est_frames = data['est_frames']
    if args.clip_min and args.clip_max:
        obs_frames = np.clip(obs_frames, args.clip_min, args.clip_max)
        est_frames = np.clip(est_frames, args.clip_min, args.clip_max)
    elif args.clip_min or args.clip_max:
        print("Both clip limits must be provided. Ignoring clipping")

    est_ts = np.round(data['timescale'], 8)
    obs_ts = np.round(data['t'], 3)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(est_ts.size):
        t_est = est_ts[i]
        j = np.searchsorted(obs_ts, t_est, side='right') - 1
        t_obs = obs_ts[j]

        fig.suptitle('Filename {}'.format(args.filename))
        im_orig = obs_frames[j, :, :].T
        im_norm = (im_orig-im_orig.min())/(im_orig.max()-im_orig.min())
        im_eq = equalize_adapthist(im_norm, kernel_size=(24, 120))
        # im_eq = equalize_hist(im_norm)
        ax1.clear()
        if args.equalize:
            ax1.imshow(im_eq, cmap='gray')
        else:
            ax1.imshow(im_orig, cmap='gray')
        ax1.title.set_text('Observed Image for {:02f}'.format(t_obs))

        im_orig = est_frames[i, :, :].T
        im_norm = (im_orig-im_orig.min())/(im_orig.max()-im_orig.min())
        im_eq = equalize_adapthist(im_norm, kernel_size=(24, 120))
        # im_eq = equalize_hist(im_norm)
        ax2.clear()
        if args.equalize:
            ax2.imshow(im_eq, cmap='gray')
        else:
            ax2.imshow(im_orig, cmap='gray')
        ax2.title.set_text('Predicted Image for {:02f}'.format(t_est))

        while not plt.waitforbuttonpress():
            continue
