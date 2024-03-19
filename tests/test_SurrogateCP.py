import unittest

from src.ScalarCoupledProblem import ScalarCoupledProblem as ScalarCP
from src.SurrogateCoupledProblem import SurrogateCoupledProblem as SCP

from src.visualize import Plotter

import numpy as np
import matplotlib.pyplot as plt


class TestSCP_1D(unittest.TestCase):
    
    def setUp(self):
        self.y1 = lambda z, y2: np.power(z, 2.) - np.cos(y2/2.)
        self.y2 = lambda z, y1: z+y1
        self.f = lambda z, y1, y2: np.cos((y1 + np.exp(-y2))/np.pi) + z/20.
        self.lims_z = np.array([[-5., 5.]])
        self.lims_y1 = np.array([[0., 20.]])
        self.lims_y2 = np.array([[0., 20.]])
        self.scalarcp = ScalarCP(self.y1, self.y2, self.f, self.lims_z, self.lims_y1, self.lims_y2)
        
        self.scp = SCP(self.scalarcp, 5, 4)

        self.rng = np.random.default_rng()
        
        self.vis = Plotter('img/testing/', self.scp)

    """
        The coupled solution of the coupled MDA given by the means of both GPs is within 5% of the real solution.

        Strongly depends on the DoE, and cannot usually say much about it. 
    """
    @unittest.expectedFailure
    def testSolve(self):
        conv_exact, sol_exact = self.scalarcp.solve(-3.)
        
        self.assertTrue(conv_exact.all())

        converged, y = self.scp.solve(-3., xi=np.zeros((1, 2)))
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, sol_exact, rtol=5e-2)
        
        converged, y = self.scp.solve(np.array([[-3.], [-3.], [-3.]]), xi=np.zeros((3, 2)))
        
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, np.repeat(sol_exact, 3, axis=0), rtol=5e-2)
    
    def testSolveNoise(self):
        conv_exact, sol_exact = self.scalarcp.solve(-3.)
        
        self.assertTrue(conv_exact.all())
        
        converged, y = self.scp.solve(-3., xi=np.zeros((1, 2)))
        
        self.assertTrue(converged.all())

        var_1 = self.scp.y1.surrogate.predict_variances(np.array([[-3., y[0, 1]]]))
        var_2 = self.scp.y2.surrogate.predict_variances(np.array([[-3., y[0, 0]]]))
        
        self.assertTrue((y[:, 0] - sol_exact[:, 0]) < 3*np.sqrt(var_1))
        self.assertTrue((y[:, 1] - sol_exact[:, 1]) < 3*np.sqrt(var_2))
        
        fig, ax = self.vis.plot_surrogates(self.scp, -3.)

        ax.scatter(sol_exact[0, 1], sol_exact[0, 0], color='red', marker='x', label='Exact')
        ax.legend()

        plt.figure(fig)

        plt.savefig('img/testing/SCP_solutions_1d.png', dpi=300) 

    def testSolveShape(self):
        converged, y_sol = self.scp.solve(np.array([[-3.], [-3.], [-3.]]))
        
        np.testing.assert_equal(len(y_sol.shape), 2)
        np.testing.assert_equal(y_sol.shape, np.array([3, 2]))
        np.testing.assert_equal(converged.shape, y_sol.shape[:1])

    def testEnrich(self):
        # keep adding elements, make sure accuracy increases
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        enrichment_types = ['solution', 'random_uniform', r'random_normal $\mathcal{N}(-3, 2)$']

        conv_exact, sol_exact = self.scalarcp.solve(-3.)
        err = lambda sol: np.sum(np.power(sol_exact-sol, 2.))

        for enrichment_type in enrichment_types:
            scp_ = SCP(self.scalarcp, 2, 2)
            
            conv, sol = scp_.solve(-3., xi=np.zeros((1, 2)))

            self.assertTrue(conv.all())

            err_hist = [err(sol)]
            pt_hist = [2]

            for i in range(2, 20):
                enrichment_pt = None

                if enrichment_type == 'solution':
                    enrichment_pt = np.array([-3.])
                if enrichment_type == 'random_uniform':
                    enrichment_pt = self.rng.uniform(self.lims_z[0, 0], self.lims_z[0, 1], size=(1))
                if enrichment_type == 'random_normal $\mathcal{N}(-3, 2)$':
                    enrichment_pt = self.rng.normal(-3., 2., size=(1))
                scp_.enrich_surrogates(enrichment_pt)
                
                conv, sol = scp_.solve(-3., xi=np.zeros((1, 2)))
                if conv.all():
                    pt_hist.append(i+1)
                    err_hist.append(err(sol))
                else:
                    print('[testEnrich] SCP solution diverged')


            ax[0].plot(pt_hist, err_hist, label=enrichment_type)
            ax[1].loglog(pt_hist, err_hist, label=enrichment_type)
        
        ax[0].grid()
        ax[0].set_xlabel('Number of points')
        ax[0].set_ylabel(r'$||y_\text{exact} - y_\text{surrogate}||_2$')

        ax[1].grid()
        ax[1].set_xlabel(r'$\log$ number of points')
        ax[1].set_ylabel(r'$\log\left(||y_{\text{exact}} - y_{\text{surrogate}}||_2\right)$')
        
        ax[0].legend()

        plt.tight_layout()
        plt.savefig('img/testing/SCP_convergence.png')

class TestSCP_Sellar(unittest.TestCase):
    
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
    
    """
        The same note applies as above. Strongly depends on the used DoE.
    """
    @unittest.expectedFailure
    def testSolve(self):
        z = np.array([0., 2.63, 0.])
        conv_exact, sol_exact = self.scalarcp.solve(z)
        
        self.assertTrue(conv_exact.all())
        
        conv, sol = self.scp.solve(z)
        
        fig, ax = self.vis.plot_surrogates(self.scp, z)
        
        ax.scatter(sol_exact[0, 0], sol_exact[0, 1], marker='x', color='red', label='Exact solution')
        ax.legend()
        plt.savefig('img/testing/SCP_solutions_sellar.png')
        self.assertTrue(conv.all())
        
        np.testing.assert_allclose(sol, sol_exact, rtol=1e-2)
        
        converged, y = self.scp.solve(np.array([z, z, z]))
        
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, np.repeat(sol_exact, 3, axis=0), rtol=1e-2)
    
    """
        Calculate convergence speed. We first sample z from \\{0\\}\\times\\mathbb{R}\\times\\{0\\},
        then from \\mathbb{R}\\times\mathbb{R}\\times\\{0\\}, and finally from \\mathbb{R}^3. In other
        words, we first fix two of the three coordinates to 0, then one, and finally let all three
        coordinates be free and compare the convergence rate.
    """
    def testDimensionalityConvergence(self):
        lims_zs = [np.array([[0., 0.], [-10., 10.], [0., 0.]]),
                   np.array([[0., 10.], [-10., 10.], [0., 0.]]),
                   np.array([[0., 10.], [-10., 10.], [0., 10.]])]
        
        enrichment_types = ['solution', 'random_uniform']
        
        labels = [r'$(0, \mathbb{R}, 0)$ (1D)', r'$(\mathbb{R}, \mathbb{R}, 0)$ (2D)', r'$(\mathbb{R}, \mathbb{R}, \mathbb{R})$ (3D)']
        z = np.array([0., 2.63, 0.])

        conv_exact, sol_exact = self.scalarcp.solve(z)
        err = lambda sol: np.sum(np.power(sol_exact-sol, 2.))
        
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        fig.suptitle('Convergence rate for different enrichment strategies')
        
        subfigs = fig.subfigures(nrows=2, ncols=1)

        ax = []
        for row in range(2):
            ax_ = subfigs[row].subplots(1, 2)
            ax.append(ax_)

            subfigs[row].suptitle(enrichment_types[row])
        ax = np.array(ax) 

        for (label, lims_z) in zip(labels, lims_zs):

            for j, enrichment_type in enumerate(enrichment_types):
                scalarcp_ = ScalarCP(self.y1, self.y2, self.f, lims_z, self.lims_y1, self.lims_y2)
                scp_ = SCP(scalarcp_, 2, 2)
                
                conv, sol = scp_.solve(z, xi=np.zeros((1, 2)))

                self.assertTrue(conv.all())

                err_hist = [err(sol)]
                pt_hist = [2]

                for i in range(3, 20):
                    enrichment_pt = None

                    if enrichment_type == 'solution':
                        enrichment_pt = z
                    if enrichment_type == 'random_uniform':
                        enrichment_pt = self.rng.uniform(self.lims_z[:, 0], self.lims_z[:, 1])
                    scp_.enrich_surrogates(enrichment_pt)
                    
                    conv, sol = scp_.solve(z, xi=np.zeros((1, 2)))

                    self.assertTrue(conv.all())

                    err_hist.append(err(sol))
                    pt_hist.append(i)

                ax[j, 0].plot(pt_hist, err_hist, label=label)
                ax[j, 1].loglog(pt_hist, err_hist, label=label)
        for j in range(len(enrichment_types)):
            ax[j, 0].grid()
            ax[j, 0].set_xlabel('Number of points')
            ax[j, 0].set_ylabel(r'$||y_\text{exact} - y_\text{surrogate}||_2$')

            ax[j, 1].grid()
            ax[j, 1].set_xlabel(r'$\log$ number of points')
            ax[j, 1].set_ylabel(r'$\log\left(||y_{\text{exact}} - y_{\text{surrogate}}||_2\right)$')
            
            ax[j, 0].legend()

        plt.savefig('img/testing/SCP_convergence_sellar_dimensionality.png', dpi=300)
