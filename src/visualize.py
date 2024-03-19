import numpy as np
import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, directory, SCP):
        self.rng = np.random.default_rng()

        self.SCP = SCP
        self.directory = directory
    
    """
        Plot EI over z_i, where i is one of the dimensions of z. The dimension i is given by z_free_dim.
        The other coordinates of z are kept fixed and are given by z_fixed.
    """
    def plot_EI(self, KL_GP, z_free_dim=0, z_fixed=None, filename='EI.png'):
        if z_fixed is None:
            z_fixed = np.zeros(self.SCP.dim_z)
        # assuming 1D problem
        z = np.linspace(self.SCP.lims_z[z_free_dim, 0], self.SCP.lims_z[z_free_dim, 1], 100)

        EI = []
        for z_ in z:
            z_sample = z_fixed
            z_sample[z_free_dim] = z_
            EI.append(KL_GP.EI(z_sample)[1])
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(z, EI)
        ax.set_xlabel('z')
        ax.set_ylabel('EI')
        
        for z_doe in KL_GP.DoE_z:
            ax.axvline(z_doe, color='red', linestyle='--')

        plt.savefig('%s/%s' % (self.directory, filename))

        return fig, ax

    # assuming one dimensional coupling variables
    def plot_surrogates(self, SCP, z, filename=None):
        x_pts = np.linspace(SCP.lims_y2[0, 0], SCP.lims_y2[0, 1], 1000)
        y_pts = np.linspace(SCP.lims_y1[0, 0], SCP.lims_y1[0, 1], 1000)
        
        if isinstance(z, float):
            z = np.array([z])
        
        pts_1 = np.concatenate((np.repeat(z.reshape(1, -1), x_pts.shape[0], axis=0), x_pts.reshape(-1, 1)), axis=1)
        pts_2 = np.concatenate((np.repeat(z.reshape(1, -1), y_pts.shape[0], axis=0), y_pts.reshape(-1, 1)), axis=1)
        
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.set_xlim(SCP.lims_y2[0, 0], SCP.lims_y2[0, 1])
        ax.set_ylim(SCP.lims_y1[0, 0], SCP.lims_y1[0, 1])
        ax.grid()
        
        mean_1 = SCP.y1.surrogate.predict_values(pts_1)
        var_1 = SCP.y1.surrogate.predict_variances(pts_1)
        ax.plot(x_pts, mean_1, linestyle='-', color='red')
        ax.plot(x_pts, mean_1 - 3*np.sqrt(var_1), linestyle='--', color='gray')
        ax.plot(x_pts, mean_1 + 3*np.sqrt(var_1), linestyle='--', color='gray')
        
        mean_2 = SCP.y2.surrogate.predict_values(pts_2)
        var_2 = SCP.y2.surrogate.predict_variances(pts_2)
        
        ax.plot(mean_2, y_pts, color='blue', label=r'$\mu$')
        ax.plot(mean_2 + 3*np.sqrt(var_2), y_pts, linestyle='--', color='black', label=r'$\mu \pm 3\sigma$')
        ax.plot(mean_2 - 3*np.sqrt(var_2), y_pts, linestyle='--', color='black')
        
        xi = self.rng.normal(size=(500, 2))
        conv, res = SCP.solve(z, xi)

        ax.scatter(*np.flip(res[conv, :].T), marker='x', color='green', label='Converged')
        ax.scatter(*np.flip(res[np.logical_not(conv), :].T), marker='x', color='orange', label='Diverged')

        if filename is not None:
            plt.savefig(filename, dpi=300)
        return fig, ax

