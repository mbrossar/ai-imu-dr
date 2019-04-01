import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
import scipy.linalg
from termcolor import cprint
from utils import *


class NUMPYIEKF:
    Id2 = np.eye(2)
    Id3 = np.eye(3)
    Id6 = np.eye(6)
    IdP = np.eye(21)

    def __init__(self, parameter_class=None):

        # variables to initialize with `filter_parameters`
        self.g = None
        self.cov_omega = None
        self.cov_acc = None
        self.cov_b_omega = None
        self.cov_b_acc = None
        self.cov_Rot_c_i = None
        self.cov_t_c_i = None
        self.cov_lat = None
        self.cov_up = None
        self.cov_b_omega0 = None
        self.cov_b_acc0 = None
        self.cov_Rot0 = None
        self.cov_v0 = None
        self.cov_Rot_c_i0 = None
        self.cov_t_c_i0 = None
        self.Q = None
        self.Q_dim = None
        self.n_normalize_rot = None
        self.n_normalize_rot_c_i = None
        self.P_dim = None
        self.verbose = None

        # set the parameters
        if parameter_class is None:
            filter_parameters = NUMPYIEKF.Parameters()
        else:
            filter_parameters = parameter_class()
        self.filter_parameters = filter_parameters
        self.set_param_attr()

    class Parameters:
        g = np.array([0, 0, -9.80665])
        """gravity vector"""

        P_dim = 21
        """covariance dimension"""

        Q_dim = 18
        """process noise covariance dimension"""

        # Process noise covariance
        cov_omega = 1e-3
        """gyro covariance"""
        cov_acc = 1e-2
        """accelerometer covariance"""
        cov_b_omega = 6e-9
        """gyro bias covariance"""
        cov_b_acc = 2e-4
        """accelerometer bias covariance"""
        cov_Rot_c_i = 1e-9
        """car to IMU orientation covariance"""
        cov_t_c_i = 1e-9
        """car to IMU translation covariance"""

        cov_lat = 0.2
        """Zero lateral velocity covariance"""
        cov_up = 300
        """Zero lateral velocity covariance"""

        cov_Rot0 = 1e-3
        """initial pitch and roll covariance"""
        cov_b_omega0 = 6e-3
        """initial gyro bias covariance"""
        cov_b_acc0 = 4e-3
        """initial accelerometer bias covariance"""
        cov_v0 = 1e-1
        """initial velocity covariance"""
        cov_Rot_c_i0 = 1e-6
        """initial car to IMU pitch and roll covariance"""
        cov_t_c_i0 = 5e-3
        """initial car to IMU translation covariance"""

        # numerical parameters
        n_normalize_rot = 100
        """timestamp before normalizing orientation"""
        n_normalize_rot_c_i = 1000
        """timestamp before normalizing car to IMU orientation"""

        def __init__(self, **kwargs):
            self.set(**kwargs)

        def set(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def set_param_attr(self):

        # get a list of attribute only
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        self.Q = np.diag([self.cov_omega, self.cov_omega, self. cov_omega,
                           self.cov_acc, self.cov_acc, self.cov_acc,
                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])

    def run(self, t, u, measurements_covs, v_mes, p_mes, N, ang0):
        dt = t[1:] - t[:-1]  # (s)
        if N is None:
            N = u.shape[0]
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(dt, u, p_mes, v_mes,
                                       ang0, N)

        for i in range(1, N):
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
                               t_c_i[i-1], P, u[i], dt[i-1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i],
                            i, measurements_covs[i])
            # correct numerical error every second
            if i % self.n_normalize_rot == 0:
                Rot[i] = self.normalize_rot(Rot[i])
            # correct numerical error every 10 seconds
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = self.normalize_rot(Rot_c_i[i])
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, ang0, N):
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.init_saved_state(dt, N, ang0)
        Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]
        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        P = np.zeros((self.P_dim, self.P_dim))
        P[:2, :2] = self.cov_Rot0*self.Id2  # no yaw error
        P[3:5, 3:5] = self.cov_v0*self.Id2
        P[9:12, 9:12] = self.cov_b_omega0*self.Id3
        P[12:15, 12:15] = self.cov_b_acc0*self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0*self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0*self.Id3
        return P

    def init_saved_state(self, dt, N, ang0):
        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        Rot_c_i = np.zeros((N, 3, 3))
        t_c_i = np.zeros((N, 3))
        Rot_c_i[0] = np.eye(3)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev,
                  t_c_i_prev, P_prev, u, dt):
        acc = Rot_prev.dot(u[3:6] - b_acc_prev) + self.g
        v = v_prev + acc * dt
        p = p_prev + v_prev*dt + 1/2 * acc * dt**2
        omega = u[:3] - b_omega_prev
        Rot = Rot_prev.dot(self.so3exp(omega * dt))
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev
        t_c_i = t_c_i_prev
        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev,
                               b_acc_prev, u, dt)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev, b_omega_prev,
                      b_acc_prev, u, dt):
        F = np.zeros((self.P_dim, self.P_dim))
        G = np.zeros((self.P_dim, self.Q_dim))
        v_skew_rot = self.skew(v_prev).dot(Rot_prev)
        p_skew_rot = self.skew(p_prev).dot(Rot_prev)

        F[3:6, :3] = self.skew(self.g)
        F[6:9, 3:6] = self.Id3
        G[3:6, 3:6] = Rot_prev
        F[3:6, 12:15] = -Rot_prev
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot
        G[9:15, 6:12] = self.Id6
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        F = F * dt
        G = G * dt
        F_square = F.dot(F)
        F_cube = F_square.dot(F)
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube
        P = Phi.dot(P_prev + G.dot(self.Q).dot(G.T)).dot(Phi.T)
        return P

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i, measurement_cov):
        # orientation of body frame
        Rot_body = Rot.dot(Rot_c_i)
        # velocity in imu frame
        v_imu = Rot.T.dot(v)
        # velocity in body frame
        v_body = Rot_c_i.T.dot(v_imu)
        # velocity in body frame in the vehicle axis
        v_body += self.skew(t_c_i).dot(u[:3] - b_omega)
        Omega = self.skew(u[:3] - b_omega)

        # Jacobian w.r.t. car frame
        H_v_imu = Rot_c_i.T.dot(self.skew(v_imu))
        H_t_c_i = -self.skew(t_c_i)

        H = np.zeros((2, self.P_dim))
        H[:, 3:6] = Rot_body.T[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]
        r = - v_body[1:]
        R = np.diag(measurement_cov)
        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        S = H.dot(P).dot(H.T) + R
        K = (np.linalg.solve(S, P.dot(H.T).T)).T
        dx = K.dot(r)

        dR, dxi = NUMPYIEKF.sen3exp(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.dot(Rot)
        v_up = dR.dot(v) + dv
        p_up = dR.dot(p) + dp

        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        dR = NUMPYIEKF.so3exp(dx[15:18])
        Rot_c_i_up = dR.dot(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        I_KH = NUMPYIEKF.IdP - K.dot(H)
        P_up = I_KH.dot(P).dot(I_KH.T) + K.dot(R).dot(K.T)
        P_up = (P_up + P_up.T)/2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def skew(x):
        X = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
        return X

    @staticmethod
    def rot_from_2_vectors(v1, v2):
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        v = np.cross(v1, v2)
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(v)
        Rot = NUMPYIEKF.Id3 + NUMPYIEKF.skew(v) + \
              NUMPYIEKF.skew(v).dot(NUMPYIEKF.skew(v))*(1-cosang)/(sinang**2)
        Rot = NUMPYIEKF.normalize_rot(Rot)
        return Rot

    @staticmethod
    def sen3exp(xi):
        phi = xi[:3]

        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = np.array([[0, -phi[2], phi[1]],
                        [phi[2], 0, -phi[0]],
                        [-phi[1], phi[0], 0]])
            J = NUMPYIEKF.Id3 + 0.5 * skew_phi
            Rot = NUMPYIEKF.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * NUMPYIEKF.Id3 \
                   + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis
            Rot = c * NUMPYIEKF.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

        x = J.dot(xi[3:].reshape(-1, 3).T)
        return Rot, x

    @staticmethod
    def so3exp(phi):
        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = np.array([[0, -phi[2], phi[1]],
                      [phi[2], 0, -phi[0]],
                      [-phi[1], phi[0], 0]])
            return np.identity(3) + skew_phi

        axis = phi / angle
        skew_axis = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        s = np.sin(angle)
        c = np.cos(angle)

        return c * NUMPYIEKF.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

    @staticmethod
    def so3left_jacobian(phi):
        """

        :param phi:
        :return:
        """

        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = np.array([[0, -phi[2], phi[1]],
                      [phi[2], 0, -phi[0]],
                      [-phi[1], phi[0], 0]])
            return NUMPYIEKF.Id3 + 0.5 * skew_phi

        axis = phi / angle
        skew_axis = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        s = np.sin(angle)
        c = np.cos(angle)

        return (s / angle) * NUMPYIEKF.Id3 \
               + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis

    @staticmethod
    def normalize_rot(Rot):

        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(Rot, full_matrices=False)

        S = np.eye(3)
        S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
        return U.dot(S).dot(V)

    @staticmethod
    def from_rpy(roll, pitch, yaw):
        return NUMPYIEKF.rotz(yaw).dot(NUMPYIEKF.roty(pitch).dot(NUMPYIEKF.rotx(roll)))

    @staticmethod
    def rotx(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]])

    @staticmethod
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    @staticmethod
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

    @staticmethod
    def to_rpy(Rot):
        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.isclose(pitch, np.pi / 2.):
            yaw = 0.
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                             Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch,
                              Rot[2, 2] * sec_pitch)
        return roll, pitch, yaw

    def set_learned_covariance(self, torch_iekf):
        torch_iekf.set_Q()
        self.Q = torch_iekf.Q.cpu().detach().numpy()

        beta = torch_iekf.initprocesscov_net.init_cov(torch_iekf)\
            .detach().cpu().numpy()

        self.cov_Rot0 *= beta[0]
        self.cov_v0 *= beta[1]
        self.cov_b_omega0 *= beta[2]
        self.cov_b_acc0 *= beta[3]
        self.cov_Rot_c_i0 *= beta[4]
        self.cov_t_c_i0 *= beta[5]
