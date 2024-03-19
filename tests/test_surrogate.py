import unittest
from src.SurrogateFunction import Surrogate as Surrogate

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import functools


from smt.sampling_methods import LHS

class TestSurrogate_1D(unittest.TestCase):
    """
        1-dimensional linear problem. Keep z constant, and test that surrogate
        modeling works for a linear 1 dimensional function.
    """
    def setUp(self):
        def y(z, y):
            return z + y
        self.y = y
        
        self.z = 3.
        self.lims_y = np.array([0., 5.])

        self.DoE_z = 3*np.ones((5, 1))
        self.DoE_coupling = np.linspace(self.lims_y[0], self.lims_y[1], 5).reshape(-1, 1)

        self.surrogate = Surrogate(self.y, self.DoE_z, self.DoE_coupling)

    def testSample(self):
        rng = np.random.default_rng()
        z_vals = self.z * np.ones((100, 1))
        y_vals = self.lims_y[0] + (self.lims_y[1]-self.lims_y[0])*rng.uniform(size=(100, 1))
        
        np.testing.assert_almost_equal(self.surrogate.sample(z_vals, y_vals, xi=np.zeros((100, 1))), self.y(z_vals, y_vals),  decimal=5)

        # test noise
        y_vals = (self.lims_y[0] - 5) + (10+self.lims_y[1]-self.lims_y[0])*rng.uniform(size=(100, 1))
        output = self.surrogate.sample(z_vals, y_vals)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_vals, output, label='Prediction')
        ax.scatter(self.DoE_coupling, self.y(self.DoE_z, self.DoE_coupling), marker='x', color='red', label='Training')

        ax.grid()
        ax.legend()
        ax.set_xlabel('y_coupling')
        ax.set_ylabel('y_output')

        plt.savefig('img/testing/surrogate.png')

    def testSolveShape(self):
        z_vals = np.array([3.])
        y_coupling_vals = np.zeros((10000, 1))
        
        y_out = self.surrogate(z_vals, y_coupling_vals)
        
        np.testing.assert_equal(y_out.shape, np.array([10000, 1]))

        z_vals = np.repeat(np.array([z_vals]), 10000, axis=0)
        y_out = self.surrogate(z_vals, y_coupling_vals)
        
        np.testing.assert_equal(y_out.shape, np.array([10000, 1]))

class TestSurrogate_cosine(unittest.TestCase):
    """
        1-dimensional nonlinear case. Keep z constant, and test that
        surrogate accurately models 1-D nonlinear function.
    """
    def setUp(self):
        def y(z, y):
            return z*np.cos(y)
        self.y = y
        
        self.z = 3.
        self.lims_y = np.array([0., np.pi])

        self.DoE_z = 3*np.ones((5, 1))
        self.DoE_coupling = np.linspace(self.lims_y[0], self.lims_y[1], 5).reshape(-1, 1)

        self.surrogate = Surrogate(self.y, self.DoE_z, self.DoE_coupling)
        
        self.rng = np.random.default_rng()

    def testSample(self):
        n_sample = 300

        z_vals = self.z * np.ones((n_sample, 1))
        y_vals = self.rng.uniform(self.lims_y[0], self.lims_y[1], size=(n_sample, 1))
        
        np.testing.assert_almost_equal(self.surrogate.sample(z_vals, y_vals, xi=np.zeros((n_sample, 1))), self.y(z_vals, y_vals), decimal=1)

        # test noise
        y_vals = self.rng.uniform(-np.pi, 2*np.pi, size=(n_sample, 1))
        output = self.surrogate.sample(z_vals, y_vals)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.axvline(0., color='gray', linestyle='--')
        ax.axvline(np.pi, color='gray', linestyle='--')
    
        ax.scatter(y_vals, output, label='Prediction')
        ax.scatter(self.DoE_coupling, self.y(self.DoE_z, self.DoE_coupling), marker='x', color='red', label='Training')
        
        y_vals = np.linspace(-np.pi, 2*np.pi, 10000).reshape(-1, 1)
        mean_vals = self.surrogate.surrogate.predict_values(np.concatenate((3.*np.ones((10000, 1)), y_vals), axis=1))
        var_vals = self.surrogate.surrogate.predict_variances(np.concatenate((3.*np.ones((10000, 1)), y_vals), axis=1))
        
        ax.plot(y_vals, mean_vals, color='red', label='mean')
        ax.plot(y_vals, mean_vals - 3*np.sqrt(var_vals), color='black', label=r'$3 \sigma$')
        ax.plot(y_vals, mean_vals + 3*np.sqrt(var_vals), color='black')
        

        ax.grid()
        ax.legend()
        ax.set_xlabel('y_coupling')
        ax.set_ylabel('y_output')

        plt.savefig('img/testing/surrogate_cosine.png')

    
    def testEnrich(self):
        surrogate_enriched = Surrogate(self.y, self.DoE_z[:2, :], self.DoE_coupling[:2, :])

        for (z_, y_) in zip(self.DoE_z[2:], self.DoE_coupling[2:]):
            surrogate_enriched.enrich(z_, y_)
        
        n_sample = 300

        z_vals = 3*np.ones((n_sample, 1))
        y_coupling_test = self.rng.uniform(0., np.pi, size=(n_sample, 1))
        
        y_out = self.surrogate.sample(z_vals, y_coupling_test, xi=np.zeros((n_sample, 1)))
        y_out_enriched = surrogate_enriched.sample(z_vals, y_coupling_test, xi=np.zeros((n_sample, 1)))

        np.testing.assert_almost_equal(y_out_enriched, y_out, decimal=5)
    
    def testConvergence(self):
        pass

class TestConvergence_2D(unittest.TestCase):
    """
        2-dimensional nonlinear case. Test that the solution of the surrogate converges
        as the number of conditioning points increases. Both z and y are varied.
    """
    def setUp(self):
        def y(z, y_coupling):
            return z*np.cos(5*y_coupling)
        self.y = y
        
        self.lims_z = np.array([0., 5.])
        self.lims_y = np.array([0., np.pi])
        
        self.sampling = LHS(xlimits=np.array([self.lims_z, self.lims_y]))
        
        # sample 200 points, but feed them to the surrogate model one by one
        self.DoE_full = self.sampling(200)
        self.DoE_z = self.DoE_full[:2, :1]
        self.DoE_coupling = self.DoE_full[:2, 1:]

        self.surrogate = Surrogate(self.y, self.DoE_z, self.DoE_coupling)
        
        self.rng = np.random.default_rng()
    
    # move to src/visualize.py
    """
        Plot the error over the plane. Used in this context to plot one frame of the error gif,
        but can be used to plot simple images as well.
    """
    def _plotError(self, err, iteration, ax):
        #fig, ax = plt.subplots(figsize=(5, 5))
        ax.clear() 
        
        img = ax.imshow(err, extent=[0., 5., 0., np.pi], interpolation=None, aspect='auto', origin='lower')
        data = ax.scatter(self.surrogate.DoE_z.flatten(), self.surrogate.DoE_coupling.flatten(), color='red', marker='x', label='Training point')
        
        ax.set_xlabel('z')
        ax.set_ylabel('y_coupling')

        ax.set_title('$L^2$ error of GP conditioned on %d points' % self.surrogate.DoE_z.shape[0])

        #plt.savefig('img/testing/surrogate_convergence/%02d_2d.png' % iteration)
        return img, data 
    def _calc_l2_error(self, err):
        return err.mean() * (self.lims_z[1]-self.lims_z[0])*(self.lims_y[1]-self.lims_y[0])
    """
        Calculates the error of the surrogate model, and plots one frame of the gif of the
        error as the surrogate is conditioned on more and more observations.
    """
    def _calcError(self, iteration, zz, yy, out_exact, ax):
        while iteration+2 > self.surrogate.DoE_z.shape[0]:
            self.surrogate.enrich(np.array([self.DoE_full[iteration+2, 0]]), np.array([self.DoE_full[iteration+2, 1]]))
        out = self.surrogate(zz.reshape(-1, 1), yy.reshape(-1, 1), xi=np.zeros((zz.size, 1)))
        err = np.power(out_exact - out, 2.)
        err = err.reshape(zz.shape[0], -1)
        
        if self.surrogate.DoE_z.shape[0] > self.npt_hist[-1]:
            self.npt_hist.append(self.surrogate.DoE_z.shape[0])
            self.err_hist.append(self._calc_l2_error(err))

        return self._plotError(err, iteration+1, ax)
    """
        Plot gif of error over the domain Z\\times C, as we keep adding more points.
        Also measure the L2 error, and plot the convergence rate.
    """
    def testConvergence(self):
        z = np.linspace(self.lims_z[0], self.lims_z[1], 100)
        y_coupling = np.linspace(self.lims_y[0], self.lims_y[1], 100)

        zz, yy = np.meshgrid(z, y_coupling)
        
        out_exact = self.y(zz.reshape(-1, 1), yy.reshape(-1, 1))
        out = self.surrogate(zz.reshape(-1, 1), yy.reshape(-1, 1), xi=np.zeros((zz.size, 1)))
        
        err = np.power(out_exact - out, 2.)
        err = err.reshape(zz.shape[0], -1)
        
        self.npt_hist = [self.surrogate.DoE_z.shape[0]]
        self.err_hist = [self._calc_l2_error(err)]

        fig, ax = plt.subplots(figsize=(5, 5))
        ani = FuncAnimation(fig, functools.partial(self._calcError, zz=zz, yy=yy, out_exact=out_exact, ax=ax), interval=300, blit=True, repeat=True, frames=30)
        ani.save("img/testing/surrogate_convergence.gif", dpi=300, writer=PillowWriter(fps=1))

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].grid()

        ax[0].plot(self.npt_hist, self.err_hist)
        
        ax[1].grid()
        ax[1].loglog(self.npt_hist, self.err_hist)
        
        ax[0].set_xlabel('Number of points $N$')
        ax[0].set_ylabel(r'$L^2$ error of GP')

        ax[1].set_xlabel(r'$\log$ number of points $\log (N)$')
        ax[1].set_ylabel(r'$\log L^2$ error of GP')

        fig.suptitle(r'$L^2$ error of GP conditioned on N points')

        plt.savefig('img/testing/surrogate_convergence/convergence_rate.png')
