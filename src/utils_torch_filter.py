import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from termcolor import cprint
from utils_numpy_filter import NUMPYIEKF
from utils import prepare_data

class InitProcessCovNet(torch.nn.Module):

        def __init__(self):
            super(InitProcessCovNet, self).__init__()

            self.beta_process = 3*torch.ones(2).double()
            self.beta_initialization = 3*torch.ones(2).double()

            self.factor_initial_covariance = torch.nn.Linear(1, 6, bias=False).double()
            """parameters for initializing covariance"""
            self.factor_initial_covariance.weight.data[:] /= 10

            self.factor_process_covariance = torch.nn.Linear(1, 6, bias=False).double()
            """parameters for process noise covariance"""
            self.factor_process_covariance.weight.data[:] /= 10
            self.tanh = torch.nn.Tanh()

        def forward(self, iekf):
            return

        def init_cov(self, iekf):
            alpha = self.factor_initial_covariance(torch.ones(1).double()).squeeze()
            beta = 10**(self.tanh(alpha))
            return beta

        def init_processcov(self, iekf):
            alpha = self.factor_process_covariance(torch.ones(1).double())
            beta = 10**(self.tanh(alpha))
            return beta


class MesNet(torch.nn.Module):
        def __init__(self):
            super(MesNet, self).__init__()
            self.beta_measurement = 3*torch.ones(2).double()
            self.tanh = torch.nn.Tanh()

            self.cov_net = torch.nn.Sequential(torch.nn.Conv1d(6, 32, 5),
                       torch.nn.ReplicationPad1d(4),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=0.5),
                       torch.nn.Conv1d(32, 32, 5, dilation=3),
                       torch.nn.ReplicationPad1d(4),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=0.5),
                       ).double()
            "CNN for measurement covariance"
            self.cov_lin = torch.nn.Sequential(torch.nn.Linear(32, 2),
                                              torch.nn.Tanh(),
                                              ).double()
            self.cov_lin[0].bias.data[:] /= 100
            self.cov_lin[0].weight.data[:] /= 100

        def forward(self, u, iekf):
            y_cov = self.cov_net(u).transpose(0, 2).squeeze()
            z_cov = self.cov_lin(y_cov)
            z_cov_net = self.beta_measurement.unsqueeze(0)*z_cov
            measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10**z_cov_net))
            return measurements_covs


class TORCHIEKF(torch.nn.Module, NUMPYIEKF):
    Id1 = torch.eye(1).double()
    Id2 = torch.eye(2).double()
    Id3 = torch.eye(3).double()
    Id6 = torch.eye(6).double()
    IdP = torch.eye(21).double()

    def __init__(self, parameter_class=None):
        torch.nn.Module.__init__(self)
        NUMPYIEKF.__init__(self, parameter_class=None)

        # mean and standard deviation of parameters for normalizing inputs
        self.u_loc = None
        self.u_std = None
        self.initprocesscov_net = InitProcessCovNet()
        self.mes_net = MesNet()
        self.cov0_measurement = None

        # modified parameters
        self.IdP = torch.eye(self.P_dim).double()

        if parameter_class is not None:
            self.filter_parameters = parameter_class()
            self.set_param_attr()

    def set_param_attr(self):
        # get a list of attribute only
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self. cov_omega,
                                           self.cov_acc, self.cov_acc, self.cov_acc,
                                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            ).double()
        self.cov0_measurement = torch.Tensor([self.cov_lat, self.cov_up]).double()

    def run(self, t, u,  measurements_covs, v_mes, p_mes, N, ang0):

        dt = t[1:] - t[:-1]  # (s)
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(dt, u, p_mes, v_mes,
                                       N, ang0)

        for i in range(1, N):
            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = \
                self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
                               t_c_i[i-1], P, u[i], dt[i-1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i,
                            u[i], i, measurements_covs[i])
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = \
                self.init_saved_state(dt, N, ang0)
            Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])
            v[0] = v_mes[0]
            P = self.init_covariance()
            return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        beta = self.initprocesscov_net.init_cov(self)
        P = torch.zeros(self.P_dim, self.P_dim).double()
        P[:2, :2] = self.cov_Rot0*beta[0]*self.Id2  # no yaw error
        P[3:5, 3:5] = self.cov_v0*beta[1]*self.Id2
        P[9:12, 9:12] = self.cov_b_omega0*beta[2]*self.Id3
        P[12:15, 12:15] = self.cov_b_acc0*beta[3]*self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0*beta[4]*self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0*beta[5]*self.Id3
        return P


    def init_saved_state(self, dt, N, ang0):
        Rot = dt.new_zeros(N, 3, 3)
        v = dt.new_zeros(N, 3)
        p = dt.new_zeros(N, 3)
        b_omega = dt.new_zeros(N, 3)
        b_acc = dt.new_zeros(N, 3)
        Rot_c_i = dt.new_zeros(N, 3, 3)
        t_c_i = dt.new_zeros(N, 3)
        Rot_c_i[0] = torch.eye(3).double()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev, t_c_i_prev,
                  P_prev, u, dt):
        Rot_prev = Rot_prev.clone()
        acc_b = u[3:6] - b_acc_prev
        acc = Rot_prev.mv(acc_b) + self.g
        v = v_prev + acc * dt
        p = p_prev + v_prev.clone() * dt + 1/2 * acc * dt**2

        omega = (u[:3] - b_omega_prev)*dt
        Rot = Rot_prev.mm(self.so3exp(omega))

        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev.clone()
        t_c_i = t_c_i_prev

        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev,
                               u, dt)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u,
                      dt):

        F = P.new_zeros(self.P_dim, self.P_dim)
        G = P.new_zeros(self.P_dim, self.Q.shape[0])
        Q = self.Q.clone()
        F[3:6, :3] = self.skew(self.g)
        F[6:9, 3:6] = self.Id3
        G[3:6, 3:6] = Rot_prev
        F[3:6, 12:15] = -Rot_prev
        v_skew_rot = self.skew(v_prev).mm(Rot_prev)
        p_skew_rot = self.skew(p_prev).mm(Rot_prev)
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot
        G[9:12, 6:9] = self.Id3
        G[12:15, 9:12] = self.Id3
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        F = F * dt
        G = G * dt
        F_square = F.mm(F)
        F_cube = F_square.mm(F)
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube
        P_new = Phi.mm(P + G.mm(Q).mm(G.t())).mm(Phi.t())
        return P_new

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i, measurement_cov):
        # orientation of body frame
        Rot_body = Rot.mm(Rot_c_i)
        # velocity in imu frame
        v_imu = Rot.t().mv(v)
        omega = u[:3] - b_omega
        # velocity in body frame
        v_body = Rot_c_i.t().mv(v_imu) + self.skew(t_c_i).mv(omega)
        Omega = self.skew(omega)
        # Jacobian in car frame
        H_v_imu = Rot_c_i.t().mm(self.skew(v_imu))
        H_t_c_i = self.skew(t_c_i)

        H = P.new_zeros(2, self.P_dim)
        H[:, 3:6] = Rot_body.t()[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]
        r = - v_body[1:]
        R = torch.diag(measurement_cov)

        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up


    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        S = H.mm(P).mm(H.t()) + R
        Kt, _ = torch.gesv(P.mm(H.t()).t(), S)
        K = Kt.t()
        dx = K.mv(r.view(-1))

        dR, dxi = TORCHIEKF.sen3exp(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.mm(Rot)
        v_up = dR.mv(v) + dv
        p_up = dR.mv(p) + dp

        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        dR = TORCHIEKF.so3exp(dx[15:18])
        Rot_c_i_up = dR.mm(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        I_KH = TORCHIEKF.IdP - K.mm(H)
        P_upprev = I_KH.mm(P).mm(I_KH.t()) + K.mm(R).mm(K.t())
        P_up = (P_upprev + P_upprev.t())/2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def skew(x):
        X = torch.Tensor([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]]).double()
        return X

    @staticmethod
    def rot_from_2_vectors(v1, v2):
        """ Returns a Rotation matrix between vectors 'v1' and 'v2'    """
        v1 = v1/torch.norm(v1)
        v2 = v2/torch.norm(v2)
        v = torch.cross(v1, v2)
        cosang = v1.matmul(v2)
        sinang = torch.norm(v)
        Rot = TORCHIEKF.Id3 + TORCHIEKF.skew(v) + \
              TORCHIEKF.skew(v).mm(TORCHIEKF.skew(v))*(1-cosang)/(sinang**2)
        return Rot

    @staticmethod
    def sen3exp(xi):
        phi = xi[:3]
        angle = torch.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).double()
            J = TORCHIEKF.Id3 + 0.5 * skew_phi
            Rot = TORCHIEKF.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]]).double()
            s = torch.sin(angle)
            c = torch.cos(angle)

            J = (s / angle) * TORCHIEKF.Id3 + (1 - s / angle) * TORCHIEKF.outer(axis, axis)\
                   + ((1 - c) / angle) * skew_axis
            Rot = c * TORCHIEKF.Id3 + (1 - c) * TORCHIEKF.outer(axis, axis) \
                 + s * skew_axis

        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    @staticmethod
    def so3exp(phi):
        angle = phi.norm()

        # Near phi==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).double()
            Xi = TORCHIEKF.Id3 + skew_phi
            return Xi
        axis = phi / angle
        skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]]).double()
        c = angle.cos()
        s = angle.sin()
        Xi = c * TORCHIEKF.Id3 + (1 - c) * TORCHIEKF.outer(axis, axis) \
             + s * skew_axis
        return Xi

    @staticmethod
    def outer(a, b):
        ab = a.view(-1, 1)*b.view(1, -1)
        return ab

    @staticmethod
    def so3left_jacobian(phi):
        angle = torch.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).double()
            return TORCHIEKF.Id3 + 0.5 * skew_phi

        axis = phi / angle
        skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]]).double()
        s = torch.sin(angle)
        c = torch.cos(angle)

        return (s / angle) * TORCHIEKF.Id3 + (1 - s / angle) * TORCHIEKF.outer(axis, axis)\
               + ((1 - c) / angle) * skew_axis

    @staticmethod
    def to_rpy(Rot):
        """Convert a rotation matrix to RPY Euler angles."""

        pitch = torch.atan2(-Rot[2, 0], torch.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if isclose(pitch, np.pi / 2.):
            yaw = pitch.new_zeros(1)
            roll = torch.atan2(Rot[0, 1], Rot[1, 1])
        elif isclose(pitch, -np.pi / 2.):
            yaw = pitch.new_zeros(1)
            roll = -torch.atan2(Rot[0, 1],  Rot[1, 1])
        else:
            sec_pitch = 1. / pitch.cos()
            yaw = torch.atan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch)
            roll = torch.atan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch)
        return roll, pitch, yaw

    @staticmethod
    def from_rpy(roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles."""

        return TORCHIEKF.rotz(yaw).mm(TORCHIEKF.roty(pitch).mm(TORCHIEKF.rotx(roll)))

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""

        c = torch.cos(t)
        s = torch.sin(t)
        return t.new([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""

        c = torch.cos(t)
        s = torch.sin(t)
        return t.new([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""

        c = torch.cos(t)
        s = torch.sin(t)
        return t.new([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

    @staticmethod
    def normalize_rot(rot):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).

        # pytorch SVD seems to be inaccurate, so just move to numpy immediately
        U, _, V = torch.svd(rot)
        S = torch.eye(3).double()
        S[2, 2] = torch.det(U) * torch.det(V)
        return U.mm(S).mm(V.t())

    def forward_nets(self, u):
        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        measurements_covs = self.mes_net(u_n, self)
        return measurements_covs

    def normalize_u(self, u):
        return (u-self.u_loc)/self.u_std

    def get_normalize_u(self, dataset):
        self.u_loc = dataset.normalize_factors['u_loc'].double()
        self.u_std = dataset.normalize_factors['u_std'].double()

    def set_Q(self):
        """
        Update the process noise covariance
        :return:
        """

        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self. cov_omega,
                                           self.cov_acc, self.cov_acc, self.cov_acc,
                                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            ).double()

        beta = self.initprocesscov_net.init_processcov(self)
        self.Q = torch.zeros(self.Q.shape[0], self.Q.shape[0]).double()
        self.Q[:3, :3] = self.cov_omega*beta[0]*self.Id3
        self.Q[3:6, 3:6] = self.cov_acc*beta[1]*self.Id3
        self.Q[6:9, 6:9] = self.cov_b_omega*beta[2]*self.Id3
        self.Q[9:12, 9:12] = self.cov_b_acc*beta[3]*self.Id3
        self.Q[12:15, 12:15] = self.cov_Rot_c_i*beta[4]*self.Id3
        self.Q[15:18, 15:18] = self.cov_t_c_i*beta[5]*self.Id3

    def load(self, args, dataset):
        path_iekf = os.path.join(args.path_temp, "iekfnets.p")
        if os.path.isfile(path_iekf):
            mondict = torch.load(path_iekf)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", 'green')
        else:
            cprint("IEKF nets NOT loaded", 'yellow')
        self.get_normalize_u(dataset)


def isclose(mat1, mat2, tol=1e-10):
    return (mat1 - mat2).abs().lt(tol)


def prepare_filter(args, dataset):
    torch_iekf = TORCHIEKF()
    torch_iekf.load(args, dataset)
    torch_iekf = TORCHIEKF()

    # set dataset parameter
    torch_iekf.filter_parameters = args.parameter_class()
    torch_iekf.set_param_attr()
    if type(torch_iekf.g).__module__ == np.__name__:
        torch_iekf.g = torch.from_numpy(torch_iekf.g).double()

    # load model
    torch_iekf.load(args, dataset)
    torch_iekf.get_normalize_u(dataset)

    iekf = NUMPYIEKF(args.parameter_class)
    iekf.set_learned_covariance(torch_iekf)
    return iekf, torch_iekf
