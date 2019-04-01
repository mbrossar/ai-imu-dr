import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def prepare_data(args, dataset, dataset_name, i, idx_start=None, idx_end=None, to_numpy=False):
    # get data
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # get start instant
    if idx_start is None:
        idx_start = 0
    # get end instant
    if idx_end is None:
        idx_end = t.shape[0]

    t = t[idx_start: idx_end]
    u = u[idx_start: idx_end]
    ang_gt = ang_gt[idx_start: idx_end]
    v_gt = v_gt[idx_start: idx_end]
    p_gt = p_gt[idx_start: idx_end] - p_gt[idx_start]

    if to_numpy:
        t = t.cpu().double().numpy()
        u = u.cpu().double().numpy()
        ang_gt = ang_gt.cpu().double().numpy()
        v_gt = v_gt.cpu().double().numpy()
        p_gt = p_gt.cpu().double().numpy()
    return t, ang_gt, p_gt, v_gt, u


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)



def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix that minimizes the distance between a set of
    registered points.

    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """


    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

