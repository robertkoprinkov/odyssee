import unittest
from src.ScalarCoupledProblem import ScalarCoupledProblem as ScalarCP

import numpy as np

class TestScalarCP_1D(unittest.TestCase):
    
    def setUp(self):
        self.y1 = lambda z, y2: np.power(z, 2.) - np.cos(y2/2.)
        self.y2 = lambda z, y1: z+y1
        self.f = lambda z, y1, y2: np.cos((y1 + np.exp(-y2))/np.pi) + z/20.
        self.lims_z = np.array([[0., 5.]])
        self.lims_y1 = np.array([[0., 20.]])
        self.lims_y2 = np.array([[0., 20.]])
        self.scp = ScalarCP(self.y1, self.y2, self.f, self.lims_z, self.lims_y1, self.lims_y2)
    
    def testSolve(self):
        sol_exact = np.array([[9.94564331, 6.94564331]])

        converged, y = self.scp.solve(-3.)
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, sol_exact)
        
        converged, y = self.scp.solve(np.array([[-3.], [-3.], [-3.]]))
        
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, np.repeat(sol_exact, 3, axis=0))


    def testSolveShape(self):
        converged, y_sol = self.scp.solve(np.array([[-3.], [-3.], [-3.]]))
        
        np.testing.assert_equal(len(y_sol.shape), 2)
        np.testing.assert_equal(y_sol.shape, np.array([3, 2]))
        np.testing.assert_equal(converged.shape, y_sol.shape[:1])

class TestScalarCP_Sellar(unittest.TestCase):
    
    def setUp(self):
        def y1(z, y2):
            return (z[:, 0] + np.power(z[:, 1], 2.) + z[:, 2] - 0.2*y2.reshape(-1, 1)[:, 0]).reshape(-1, 1)
        def y2(z, y1):
            return (np.sqrt(y1.reshape(-1, 1)[:, 0]) + z[:, 0] + z[:, 1]).reshape(-1, 1)
        def f(z, y1, y2):
            return (z[:, 0] + np.power(z[:, 2], 2) + np.exp(-y2.reshape(-1, 1)[:, 0]) + 10*np.cos(z[:, 1])).reshape(-1, 1)
        self.y1 = y1
        self.y2 = y2
        self.f = f
        
        self.lims_z = np.array([[0., 10.], [-10., 10.], [0., 10]])
        self.lims_y1 = np.array([[1., 50.]])
        self.lims_y2 = np.array([[-1., 24.]])
        self.scp = ScalarCP(self.y1, self.y2, self.f, self.lims_z, self.lims_y1, self.lims_y2)
    
    def testSolve(self):
        sol_exact = np.array([[0., 0.]])

        converged, y = self.scp.solve(np.array([0., 0., 0.]))
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, sol_exact)
        
        converged, y = self.scp.solve(np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
        
        self.assertTrue(converged.all())
        np.testing.assert_allclose(y, np.repeat(sol_exact, 3, axis=0))


    def testSolveShape(self):
        converged, y_sol = self.scp.solve(np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
        
        np.testing.assert_equal(len(y_sol.shape), 2)
        np.testing.assert_equal(y_sol.shape, np.array([3, 2]))
        np.testing.assert_equal(converged.shape, y_sol.shape[:1])

