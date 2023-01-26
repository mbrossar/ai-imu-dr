from NLR_iekf import NUMPYIEKF as IEKF
from NLR_torch_iekf import TORCHIEKF
import pickle
import numpy as np
import torch
import time
import os
from NLR_utils import prepare_data, generate_normalize_u_p
from NLR_utils_plot import results_filter

def launch(args):
    # if args.train_filter:
    #     train_filter(args, dataset)

    if args.test_filter:
        test_filter(args)

    if args.results_filter:
        results_filter(args)

class KITTIParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))

def test_filter(args):
    t, pose0, p_gt, v0, u = prepare_data()
    u_t = torch.from_numpy(u).double()
    # generate_normalize_u_p(args, u_t)

    iekf = IEKF()
    torch_iekf = TORCHIEKF()

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args)
    iekf.set_learned_covariance(torch_iekf)

    
    N = None
    measurements_covs = torch_iekf.forward_nets(u_t)
    measurements_covs = measurements_covs.detach().numpy()
    start_time = time.time()
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs, v0, N, pose0)
    diff_time = time.time() - start_time
    print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))
    mondict = {
        't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
        'Rot_c_i': Rot_c_i, 't_c_i': t_c_i, 'measurements_covs': measurements_covs,
        }
    with open(os.path.join(args.path_results, "test_filter.p"), 'wb') as f:
        pickle.dump(mondict, f)

class KITTIArgs():
    # path_data_base = "/media/mines/46230797-4d43-4860-9b76-ce35e699ea47/KITTI/raw"
    # path_data_save = "../data"
    path_results = "../results"
    path_temp = "../temp"
    file_normalize_factor = "normalize_factors.p"

    # epochs = 400
    # seq_dim = 6000

    # # training, cross-validation and test dataset
    # cross_validation_sequences = ['2011_09_30_drive_0028_extract']
    # test_sequences = ['2011_09_30_drive_0028_extract']
    continue_training = False

    # # choose what to do
    read_data = 0
    train_filter = 0
    test_filter = 1
    results_filter = 1
    # dataset_class = KITTIDataset
    parameter_class = KITTIParameters

if __name__ == '__main__':
    launch(KITTIArgs)


    # import matplotlib.pyplot as plt

    # plt.plot(p[:,0],p[:,1])
    # plt.show()