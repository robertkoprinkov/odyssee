import numpy as np
from smt.surrogate_models import KRG

class Surrogate():
    """
        Initialize surrogate model
        @param f_original Function that takes an np.ndarray of size self.input_dim = DoE_y[1]
        @param DoE        Points to condition surrogate model on. np.ndarray of shape 
                          (n_train, self.input_dim)
        @param f_DoE      (Optional) Values that f takes on the points DoE. 
                          np.ndarray of shape (n_train, self.output_dim)
        @param output_dim Output dimension. Only support output_dim=1 for now.
    """

    def __init__(self, f_original, DoE_z, DoE_coupling, f_DoE=None, output_dim=1):
        self.f_original = f_original
        self.DoE_z = DoE_z
        self.DoE_coupling = DoE_coupling
   
        self.input_dim_z = self.DoE_z.shape[1]
        self.input_dim_coupling = self.DoE_coupling.shape[1]
        self.input_dim = self.input_dim_z + self.input_dim_coupling
        self.output_dim = output_dim
        
        self.rng = np.random.default_rng()

        if f_DoE is None:
            f_DoE = f_original(DoE_z, DoE_coupling)
        self.f_DoE = f_DoE

        DoE = np.concatenate((self.DoE_z, self.DoE_coupling), axis=1)

        self.surrogate = KRG(theta0=[1e-2]*np.ones(self.input_dim), print_prediction=False, print_global=False)
        
        self.surrogate.set_training_values(DoE, f_DoE)
        self.surrogate.train()
    
    def __call__(self, z, y_coupling, xi=None):
        return self.sample(z, y_coupling, xi)

    """
        Sample from the surrogate model. 
        @param z          Z values in space to sample at. np.ndarray of size (n_samples, self.input_dim_z)
                          or of size (self.input_dim_z,), in which case the samples are assumed to be
                          sampled at the same z value but different coupling values. The number of samples
                          is then determined from y_coupling.
        @param y_coupling Values of the coupling variables corresponding to the z values. np.ndarray of size
                          (n_samples, self.input_dim_coupling), or of size (self.input_dim_coupling), in which 
                          case the number of samples is assumed to be 1.
        @param xi         Seeds for the sampling. If None, seeds are drawn from the normal distribution.

        @return np.ndarray of size (n_samples, self.output_dim), or of size (self.output_dim) if the
                input was of size (self.input_dim,)
    """
    def sample(self, z, y_coupling, xi=None):
        if len(z.shape) == 1 and len(y_coupling) != 1:
            z = np.repeat(z, y_coupling.shape[0], axis=0)
        
        assert(z.shape[0] == y_coupling.shape[0])
        assert(z.shape[1] == self.input_dim_z and y_coupling.shape[1] == self.input_dim_coupling)
        
        x = np.concatenate((z, y_coupling), axis=1)
        
        mean = self.surrogate.predict_values(x)
        var  = self.surrogate.predict_variances(x)
        
        if xi is None:
            xi = self.rng.normal(size=(z.shape[0], 1))
        
        return mean + xi.reshape(z.shape[0], 1)*np.sqrt(var)
    
    """
        Enrich surrogate with the value of the underlying solver at one point

        @param z Z
        @param y_coupling Values of the coupling variables
    """
    def enrich(self, z, y_coupling):
        # enrich surrogate with one point
        self.DoE_z = np.concatenate((self.DoE_z, np.array([z])), axis=0)
        self.DoE_coupling = np.concatenate((self.DoE_coupling, np.array([y_coupling])), axis=0)
        
        self.f_DoE = np.concatenate((self.f_DoE, self.f_original(np.array([z]), y_coupling)), axis=0)

        DoE = np.concatenate((self.DoE_z, self.DoE_coupling), axis=1)
        
        self.surrogate.set_training_values(DoE, self.f_DoE)
        self.surrogate.train()
               
