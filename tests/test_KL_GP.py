import unittest

from src.ScalarCoupledProblem import ScalarCoupledProblem as ScalarCP
from src.SurrogateCoupledProblem import SurrogateCoupledProblem as SCP
from src.KL_GP import KL_GP as KL_GP
from src.PCE import PCE as PCE

from src.visualize import Plotter

import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from scipy import stats

from tqdm import tqdm

class TestKL_GP_1D(unittest.TestCase):
    
    def setUp(self):
        self.y1 = lambda z, y2: np.power(z, 2.) - np.cos(y2/2.)
        self.y2 = lambda z, y1: z+y1
        self.f = lambda z, y1, y2: np.cos((y1 + np.exp(-y2))/np.pi) + z/20.
        self.lims_z = np.array([[-5., 5.]])
        self.lims_y1 = np.array([[0., 20.]])
        self.lims_y2 = np.array([[0., 20.]])
        self.scalarcp = ScalarCP(self.y1, self.y2, self.f, self.lims_z, self.lims_y1, self.lims_y2)
        
        self.scp = SCP(self.scalarcp, 5, 4)
        
        self.DoE_z = np.array([-4.5, -2.3, 0.5, 3.5 ])
        self.order = 3

        self.klgp = KL_GP(self.DoE_z, self.scp, self.order)

        self.rng = np.random.default_rng()
        
        self.vis = Plotter('img/testing/', self.scp)
    
    """
        Test that all member variables and return variables have the expected shapes.
    """
    def testShape(self):
        np.testing.assert_equal(self.klgp.DoE_z.shape, np.array([4, 1]))
        np.testing.assert_equal(self.klgp.A.shape, np.array([4, 10]))

        # phi_GP contains one GP for the mean term, and one for each included basis vector, of which
        # there are self.klgp.truncation_index, so that the last term of phi to be included is 
        # self.klgp.phi[self.klgp.truncation_index-1], which corresponds to GP 
        # self.klgp.phi_GP[self.klgp.truncation_index]
        np.testing.assert_equal(len(self.klgp.phi_GP), self.klgp.truncation_index+1)
        
        np.testing.assert_equal(self.klgp.phi.shape, np.array([4, 4]))
        
        n_samples = 10000
        samples = self.klgp.sample(np.array([self.DoE_z[0]]), n_samples=n_samples)

        np.testing.assert_equal(samples.shape, np.array([n_samples, 1]))
        xi, mean, sigma, mean_intermediate, sigma_intermediate = self.klgp.getGaussianStatistics(np.array([self.DoE_z[0]]), n_samples=n_samples, return_intermediate=True)
        
        np.testing.assert_equal(xi.shape, np.array([n_samples, 2]))
        np.testing.assert_equal(sigma.shape, np.array([n_samples]))
        np.testing.assert_equal(mean.shape, np.array([n_samples]))
        np.testing.assert_equal(mean_intermediate.shape, np.array([self.klgp.truncation_index+1, n_samples]))
        np.testing.assert_equal(sigma_intermediate.shape, np.array([self.klgp.truncation_index+1, n_samples]))

        xi, EI = self.klgp.EI(np.array([self.DoE_z[0]]))
        np.testing.assert_equal(xi.shape, np.array([n_samples, 2]))
        np.testing.assert_equal(EI.shape, np.array([]))
        
        z, f = self.klgp.z_next(n_starting_points=2, log=False)
        np.testing.assert_equal(z.shape, np.array([self.scp.dim_z]))
        self.assertTrue(isinstance(f, float))

        DoE_z, Pmin = self.klgp.Pmin()
        np.testing.assert_equal(DoE_z.shape, self.klgp.DoE_z.shape)
        np.testing.assert_equal(Pmin.shape, np.array([self.klgp.DoE_z.shape[0]]))



    """
        Test that the PCE at DoE_z is the same when obtained through getPCE, and when calculated during the
        KL_GP construction.
    """
    def testPCE(self):
        for i, z in enumerate(self.klgp.DoE_z):
            PCE_exact = self.klgp.getPCE(z)
            PCE_klgp = self.klgp.PCE_DoE[i]
            
            np.testing.assert_equal(PCE_exact.order, PCE_klgp.order)
            
            np.testing.assert_equal(PCE_klgp.hermite_coefficients.shape, PCE_exact.hermite_coefficients.shape)
            np.testing.assert_allclose(PCE_klgp.hermite_coefficients, PCE_exact.hermite_coefficients, rtol=1e-3)
        
        # plot Y_exact, by
        # 1. Calculating PCE at every point
        # 2. Interpolating a PCE at every point with the getPCE function (assuming eta=0
        #        => taking the mean of the KL GP)
        # 3. By calculating the mean and standard deviation of the KL GP
        z = np.linspace(self.lims_z[0, 0], self.lims_z[0, 1], 100)
        
        Y_exact = []
        
        xi = self.rng.normal(size=(10000, 2))
        P = stats.norm.pdf(xi[:, 0]) * stats.norm.pdf(xi[:, 1])
        Y_klgp_pce = []
        Y_klgp_mean = []
        Y_klgp_sigma = []
        for z_ in z:
            pce = PCE(np.array([z_]), self.scp, self.order, N=100)
            pce_klgp = self.klgp.getPCE(z_, eta=np.zeros(self.klgp.truncation_index+1))
            
            # this is not correct
            # use quadrature rule
            Y_exact.append(np.sum(pce(xi[:, 0], xi[:, 1]) * P))
            
            Y_klgp_pce.append(np.sum(pce_klgp(xi[:, 0], xi[:, 1]) * P))
            
            _, mean, sigma = self.klgp.getGaussianStatistics(np.array([z_]), xi=xi)

            Y_klgp_mean.append(np.sum(mean * P))
            Y_klgp_sigma.append(np.sum(sigma * P))

        Y_exact = np.array(Y_exact)
        Y_klgp_pce = np.array(Y_klgp_pce)
        Y_klgp_mean = np.array(Y_klgp_mean)
        Y_klgp_sigma = np.array(Y_klgp_sigma)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot(z, Y_exact, label=r'$\hat{f}_{\text{obj}}(\Xi)$')
        ax.plot(z, Y_klgp_mean, label=r'$\mu_{\text{KLGP}} (\Xi)$', color='red')
        ax.plot(z, Y_klgp_mean + 3*Y_klgp_sigma, linestyle='--', color='gray', label=r'$\mu_{\text{KLGP}} (\Xi) \pm 3\sigma_{\text{KLGP}} (\Xi)$')
        ax.plot(z, Y_klgp_mean - 3*Y_klgp_sigma, linestyle='--', color='gray')
        
        ax.plot(z, Y_klgp_pce, label='KLGP (PCE)', linestyle='--')
        
        ax.set_xlabel('z')
        ax.set_ylabel(r'$Y_{\text{obj}}$')
        ax.grid()
        ax.legend()

        plt.tight_layout()
        plt.savefig('img/testing/KLGP_interpol.png', dpi=300)
        
    """
        Make sure that the vectorized calculation returns the same values as the regular one.
    """
    def testGaussianStatisticsVectorized(self):
        pass
    
    """
        Test that the mean and variance returned by getGaussianStatistics matches the one obtained through
        monte carlo estimation.
    """
    def testGaussianStatistics(self):
        z = self.rng.uniform(self.lims_z[:, 0], self.lims_z[:, 1], size=(10))

        for z_ in z:
            # we need a high number of samples for the monte carlo estimation to converge
            n_samples = 100000
            samples = self.klgp.sample(z_, n_samples=n_samples)
            _, mean, sigma = self.klgp.getGaussianStatistics(z_, n_samples=n_samples)
            
            np.testing.assert_equal(samples.shape, (n_samples, 1))
            
            np.testing.assert_allclose(samples.mean(), mean.mean(), rtol=0.15)
            np.testing.assert_allclose(np.std(samples, ddof=1), sigma.mean(), rtol=0.15)
    """
        Test that EI calculated with the Gaussian statistics and with monte carlo sampling is the same
    """
    def testEI(self):
        pass
    

class TestKL_GP_Sellar(unittest.TestCase):
    
    def setUp(self):
        def y1(z, y2):
            return (z[:, 0] + np.power(z[:, 1], 2.) + z[:, 2] - 0.2*y2.flatten()).reshape(-1, 1)
        def y2(z, y1):
            return (np.sqrt(y1.flatten()) + z[:, 0] + z[:, 1]).reshape(-1, 1)
        def f(z, y1, y2):
            return (z[:, 0] + np.power(z[:, 2], 2) + y1.flatten() + np.exp(-y2.flatten()) + 10*np.cos(z[:, 1])).reshape(-1, 1)
        self.y1 = y1
        self.y2 = y2
        self.f = f
        
        self.lims_z = np.array([[0., 10.], [-10., 10.], [0., 10]])
        
        self.lims_y1 = np.array([[1., 50.]])
        self.lims_y2 = np.array([[-1., 24.]])
        self.scalarcp = ScalarCP(self.y1, self.y2, self.f, self.lims_z, self.lims_y1, self.lims_y2)
        self.scp = SCP(self.scalarcp, 5, 5)
        
        self.vis = Plotter('img/testing/', self.scp)
        
        self.rng = np.random.default_rng()
        
        self.sampling_z = LHS(xlimits=self.lims_z)
        self.DoE_z = self.sampling_z(20)
        self.order = 3

        self.klgp = KL_GP(self.DoE_z, self.scp, self.order, truncated=False)
    
    """
        Test that the truncated series describes the same behavior as the full one.
    
    """
    def testTruncation(self):
        self.assertTrue(not self.klgp.truncated)
        self.assertTrue((self.klgp.truncation_index > 1) and (self.klgp.truncation_index <= self.klgp.phi.shape[0]))
        self.assertTrue(self.klgp.covered[self.klgp.truncation_index-1])
        
        print(self.klgp.phi_GP_energy)
        
        # add some points on the line (0, z, 0) to ensure that the variance goes to 0 at points in DoE_z
        pts_z = self.rng.uniform(self.lims_z[1, 0], self.lims_z[1, 1], size=(5))
        for pt in pts_z:
            self.klgp.add_z(np.array([0., pt, 0.]))
        # plot mean and standard deviation of the KL GP along the line (0, z, 0), 
        # taking into account the first k terms, k=1...20
        z_sample = np.linspace(self.lims_z[1, 0], self.lims_z[1, 1], 25)
         
        mean_trunc = []
        sigma_trunc = []
        f_exact = []
        f_scp = []
        
        for z in tqdm(z_sample):
            xi = self.rng.normal(size=(10000, 2))
            
            z = np.array([0., z, 0.])
            _, mean, sigma, mean_intermediate, sigma_intermediate = self.klgp.getGaussianStatistics(z, return_intermediate=True)
             
            mean_trunc.append(mean_intermediate.mean(axis=1))
            sigma_trunc.append(sigma_intermediate.mean(axis=1))
            
            conv, y_ex = self.klgp.SCP.ScalarCP.solve(z)
            y_ex = y_ex[conv, :]
            f_exact.append(self.f(np.array([z]), y_ex[0, 0], y_ex[0, 1]).mean())
            
            conv, y_scp = self.klgp.SCP.solve(z, xi=xi)
            y_scp = y_scp[conv, :]
            f_scp.append(self.f(np.repeat(np.array([z]), y_scp.shape[0], axis=0), y_scp[:, 0], y_scp[:, 1]).mean())
        mean_trunc = np.array(mean_trunc)
        sigma_trunc = np.array(sigma_trunc)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[4, 3])
        
        def to_color(r, g, b): # convert r, g, b \in [0, 1] to hex code
            color = '#'

            for intensity in [r, g, b]:
                intensity = int(np.round(intensity * 255))
                hx = hex(intensity)[2:]

                if len(hx) == 1:
                    hx = '0' + hx
                color = color + hx
            return color
        
        # if it was truncated, this is where it would be
        trunc_index = np.arange(self.klgp.covered.shape[0])[self.klgp.covered].min() + 1
         
        alpha = None
        color = None
        for n_terms in range(mean_trunc.shape[1]):
            label_mean = None
            label_std = None
            linewidth = 0.5

            if trunc_index > n_terms: # all terms up to now included, color green
                alpha = n_terms/trunc_index
                color = to_color(0, alpha, 0)

                if n_terms == trunc_index-1:
                    label_mean = r'Included: $\mu_{\text{KLGP}}(z, \Xi)$'
                    label_std = r'$\sigma_{\text{KLGP}}$'
            
            if trunc_index == n_terms: # comprises all included terms, color black
                color = 'black'
                label_mean = r'Truncation limit: $\mu_{\text{KLGP}}(z, \Xi)$'
                label_std = r'$\sigma_{\text{KLGP}}$'
                linewidth = 2.

            if trunc_index < n_terms: # contains neglected terms, color red
                alpha = (mean_trunc.shape[1] - n_terms)/(mean_trunc.shape[1]-trunc_index)
                color = to_color(alpha, 0, 0)

                if trunc_index == n_terms - 1:
                    label_mean = r'Truncated: $\mu_{\text{KLGP}}(z, \Xi)$'
                    label_std = r'$\sigma_{\text{KLGP}}(z, \Xi)$'

            alpha = np.power(alpha, 20)
            ax[0].plot(z_sample, mean_trunc[:, n_terms], color=color, linewidth = linewidth, label=label_mean)
            ax[0].plot(z_sample, mean_trunc[:, n_terms] + 3*sigma_trunc[:, n_terms], linestyle='--', color=color, linewidth=linewidth, label=label_std)
            ax[0].plot(z_sample, mean_trunc[:, n_terms] - 3*sigma_trunc[:, n_terms], linestyle='--', color=color, linewidth=linewidth)

        # plot sampling points, with the intensity of the line given by the distance from
        # the line (0, z, 0)
        def getRelDist(z):
            distance = np.sqrt(np.power(z[0], 2.) + np.power(z[1], 2.))
            max_dist = np.sqrt(np.power(self.klgp.DoE_z[:, 0], 2.) + np.power(self.klgp.DoE_z[:, 1], 2.)).max()
            dist = distance / max_dist
            
            return dist

        def getAlpha(z):
            return np.power(1-getRelDist(z), 5.)
        
        ax[0].grid()
        ax[0].set_xlabel(r'$z_1$')
        ax[0].set_ylabel(r'$\hat{Y}_{\text{obj}} (z)$')

        for z in self.klgp.DoE_z:
            ax[0].axvline(z[1], color='gray', linestyle='--', alpha=getAlpha(z))
        
        #ax[0].plot(z_sample, f_scp, label=r'$f_{\text{SCP}}(z, \Xi)$', linestyle='--')
        ax[0].plot(z_sample, f_exact, label=r'$f_\text{Exact}(z)$', linestyle='-')

        # get the legend of ax[0]. We will plot it in two columns, and to make sure the
        # legend elements align correctly we must transpose them first.
        lines = ax[0].get_legend_handles_labels()
        # calculate permutation
        reorder = np.array([[0, 1], [2, 3], [4, 5], [6, 7]]).T.flatten()
        # permute the handles and labels so they are plotted in the correct order
        handles = [lines[0][reorder[i]] for i in range(len(lines[0]))]
        labels  = [lines[1][reorder[i]] for i in range(len(lines[1]))]
        ax[1].legend(handles, labels, ncol=2, loc='center right')
        ax[1].axis('off')
        
        fig.suptitle(r'$\mu_{\text{KLGP}}$ and $\sigma_{\text{KLGP}}$ for different truncation cutoffs')

        plt.tight_layout()
        plt.savefig('img/testing/KLGP_sellar_truncation.png', dpi=300)

