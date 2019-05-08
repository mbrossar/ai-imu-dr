import matplotlib
import os
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from utils import *
from utils_torch_filter import TORCHIEKF

def results_filter(args, dataset):

    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = dataset.get_estimates(
            dataset_name)

        # get data
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        # get data for nets
        u_normalized = dataset.normalize(u).numpy()
        # shift for better viewing
        u_normalized[:, [0, 3]] += 5
        u_normalized[:, [2, 5]] -= 5

        t = (t - t[0]).numpy()
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 0] = pitch.numpy()
            ang[j, 0] = yaw.numpy()
        # unwrap
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 0] = pitch.numpy()
            ang_gt[j, 0] = yaw.numpy()

        Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        v_norm /= np.max(v_norm)

        # Compute various errors
        error_p = np.abs(p_gt - p)
        # MATE
        mate_xy = np.mean(error_p[:, :2], 1)
        mate_z = error_p[:, 2]

        # CATE
        cate_xy = np.cumsum(mate_xy)
        cate_z = np.cumsum(mate_z)

        # RMSE
        rmse_xy = 1 / 2 * np.sqrt(error_p[:, 0] ** 2 + error_p[:, 1] ** 2)
        rmse_z = error_p[:, 2]

        RotT = torch.from_numpy(Rot).float().transpose(-1, -2)

        v_r = (RotT.matmul(torch.from_numpy(v).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_gt = (Rot_gt.transpose(-1, -2).matmul(
            torch.from_numpy(v_gt).float().unsqueeze(-1)).squeeze()).numpy()

        p_r = (RotT.matmul(torch.from_numpy(p).float().unsqueeze(-1)).squeeze()).numpy()
        p_bis = (Rot_gt.matmul(torch.from_numpy(p_r).float().unsqueeze(-1)).squeeze()).numpy()
        error_p = p_gt - p_bis

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)

        # position, velocity and velocity in body frame
        fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # orientation, bias gyro and bias accelerometer
        fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        fig4, ax4 = plt.subplots(figsize=(20, 10))
        #  Measurement covariance in log scale and normalized inputs
        fig5, axs5 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # input: gyro, accelerometer
        fig6, axs6 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
        # errors: MATE, CATE  RMSE
        fig7, axs7 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        axs1[0].plot(t, p_gt)
        axs1[0].plot(t, p)
        axs1[1].plot(t, v_gt)
        axs1[1].plot(t, v)
        axs1[2].plot(t, v_r_gt)
        axs1[2].plot(t, v_r)

        axs2[0].plot(t, ang_gt)
        axs2[0].plot(t, ang)
        axs2[1].plot(t, b_omega)
        axs2[2].plot(t, b_acc)

        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.axis('equal')
        ax4.plot(p_gt[:, 0], p_gt[:, 1])
        ax4.plot(p_align[:, 0], p_align[:, 1])
        ax4.axis('equal')

        axs5[0].plot(t, np.log10(measurements_covs))
        axs5[1].plot(t, u_normalized[:, :3])
        axs5[2].plot(t, u_normalized[:, 3:])

        axs6[0].plot(t, u[:, :3])
        axs6[1].plot(t, u[:, 3:6])

        axs7[0].plot(t, mate_xy)
        axs7[0].plot(t, mate_z)
        axs7[0].plot(t, rmse_xy)
        axs7[0].plot(t, rmse_z)
        axs7[1].plot(t, cate_xy)
        axs7[1].plot(t, cate_z)
        axs7[2].plot(t, error_p)

        axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
        axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
        axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                    title="Velocity in body frame")
        axs2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                    title="Orientation")
        axs2[1].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{\omega}}$ (rad/s)',
                    title="Bias gyro")
        axs2[2].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{a}}$ (m/$\mathrm{s}^2$)',
                    title="Bias accelerometer")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        axs5[0].set(xlabel='time (s)', ylabel=r' $\mathrm{cov}(\mathbf{y}_{n})$ (log scale)',
                     title="Covariance on the zero lateral and vertical velocity measurements (log "
                           "scale)")
        axs5[1].set(xlabel='time (s)', ylabel=r'Normalized gyro measurements',
                     title="Normalized gyro measurements")
        axs5[2].set(xlabel='time (s)', ylabel=r'Normalized accelerometer measurements',
                   title="Normalized accelerometer measurements")
        axs6[0].set(xlabel='time (s)', ylabel=r'$\omega^x_n, \omega^y_n, \omega^z_n$ (rad/s)',
                    title="Gyrometer")
        axs6[1].set(xlabel='time (s)', ylabel=r'$a^x_n, a^y_n, a^z_n$ (m/$\mathrm{s}^2$)',
                    title="Accelerometer")
        axs7[0].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}_{n}-\hat{\mathbf{p}}_{n} ||$ (m)',
                    title="Mean Absolute Trajectory Error (MATE) and Root Mean Square Error (RMSE)")
        axs7[1].set(xlabel='time (s)',
                    ylabel=r'$\Sigma_{i=0}^{n} || \mathbf{p}_{i}-\hat{\mathbf{p}}_{i} ||$ (m)',
                    title="Cumulative Absolute Trajectory Error (CATE)")
        axs7[2].set(xlabel='time (s)', ylabel=r' $\mathbf{\xi}_{n}^{\mathbf{p}}$',
                    title="$SE(3)$ error on position")

        for ax in chain(axs1, axs2, axs5, axs6, axs7):
            ax.grid()
        ax3.grid()
        ax4.grid()
        axs1[0].legend(
            ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
        axs1[1].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs1[2].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs2[0].legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
                        r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$'])
        axs2[1].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        axs2[2].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed'])
        ax4.legend(['ground-truth trajectory', 'proposed'])
        axs5[0].legend(['zero lateral velocity', 'zero vertical velocity'])
        axs6[0].legend(['$\omega_n^x$', '$\omega_n^y$', '$\omega_n^z$'])
        axs6[1].legend(['$a_n^x$', '$a_n^y$', '$a_n^z$'])
        if u.shape[1] > 6:
            axs6[2].legend(['$m_n^x$', '$m_n^y$', '$m_n^z$'])
        axs7[0].legend(['MATE xy', 'MATE z', 'RMSE xy', 'RMSE z'])
        axs7[1].legend(['CATE xy', 'CATE z'])

        # save figures
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, ]
        figs_name = ["position_velocity", "orientation_bias", "position_xy", "position_xy_aligned",
                     "measurements_covs", "imu", "errors", "errors2"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + ".png"))

        plt.show(block=True)



