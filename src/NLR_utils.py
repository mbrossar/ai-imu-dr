import pickle
import numpy as np
import os

def prepare_data():
    with open('t.p', 'rb') as f:
        t = pickle.load(f)
    with open('u.p', 'rb') as f:
        u = pickle.load(f)
    u[:,3:]*=9.80655
    ## TODO: get gps locations for p
    with open('gps.p', 'rb') as f:
        gps = pickle.load(f)
    p = gps[:,-3:]
    # p = np.zeros((len(t), 3))

    pose0 = np.zeros(3)
    v0 = np.zeros(3)
    
    
    return t, pose0, p, v0, u

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

def get_normalize_u(args):
    path_normalize_factor = os.path.join(args.path_temp, args.file_normalize_factor)
    if os.path.isfile(path_normalize_factor):
        print("Normalize factors loaded.")
        with open(path_normalize_factor, 'rb') as f:
            normalize_factors = pickle.load(f)
        u_loc = normalize_factors['u_loc'].double()
        u_std = normalize_factors['u_std'].double()
        return u_loc, u_std
    else:
        print('File ', path_normalize_factor, ' does not exist')
        quit()

def generate_normalize_u_p(args, u):
    path_normalize_factor = os.path.join(args.path_temp, args.file_normalize_factor)
    if os.path.isfile(path_normalize_factor):
        print('File ', path_normalize_factor, ' exists. Replacing it now...')
    
    # mean
    num_data = u.shape[0]
    u_loc = u.sum(dim=0)
    u_loc = u_loc / num_data

    # standard deviation
    u_std = ((u - u_loc) ** 2).sum(dim=0)
    u_std = (u_std / num_data).sqrt()
    normalize_factors = {
        'u_loc': u_loc, 'u_std': u_std,
    }

    # store in pickle
    with open(path_normalize_factor, 'wb') as f:
        pickle.dump(normalize_factors, f)

    




