import numpy as np
from scipy.linalg import eig, inv,cholesky

class ornstein_uhlenbeck_process:
    '''
    Class used for one-dimensional and multi-dimensional ornstein uhlenbeck simulations:
    Based upon the procedure described in Monte Carlo Methods in Financial Engineering by Paul Glasserman
    '''
    def __init__(self,theta, mu, sigma, delta_t, X0=None):
        self.multidim = True
        self.theta    = theta
        self.mu       = mu
        self.sigma    = sigma
        self.delta_t  = delta_t
        self.X0       = X0
        self.N        = len(self.mu)
        self.X0       = np.zeros(self.N) if X0 is None else X0

        # diagonalize the matrix theta
        eigs, self.V  = eig(theta)
        self.V_inv    = inv(self.V)
        self.Lambda   = np.diag(eigs.real)

        #assert self.V.dot(self.Lambda.dot(self.V_inv)) == self.theta, 'Theta might not be a diagonizable matrix'
        
        #prepare the other values, for computing Yt
        self.b   = self.V.dot(mu) 
        VD       = self.V.dot(sigma)
        self.Var = VD.dot(np.transpose(VD))
        #self.L   = np.transpose(cholesky(self.Var)) not necessary currently

        #initialise the values at t=0
        if X0 == None:
            self.Yt = np.transpose(np.zeros(len(self.b)))
        else:
            self.Yt = self.V.dot(X0)
        self.t  = 0
    
    def reset(self):
        #initialise the values at t=0
        if self.X0 is None:
            self.X0 = np.zeros(self.N)
            self.Yt = np.transpose(np.zeros(len(self.b)))
        else:
            self.Yt = self.V.dot(self.X0)
        self.t  = 0

        
    def step(self):
        '''
        get X(t + delta t) from X(t) using exact monte carlo simulation
        '''
        self.t += self.delta_t
        eps = np.random.multivariate_normal(np.zeros(len(self.mu)),self.Var)
        for i in range(self.N):
            self.Yt[i] = self.Yt[i] * np.exp(-self.Lambda[i,i]*self.delta_t) + (1 - np.exp(-self.Lambda[i,i]*self.delta_t) ) *  \
                         self.b[i] + np.sqrt((1 - np.exp(-2*self.Lambda[i,i]*self.delta_t)) / (2*self.Lambda[i,i]))* eps[i]
            
        return self.V_inv.dot(self.Yt)
    
    def expected_val(self):
        '''
        get EXP[X(t + delta t)] from X(t) 
        '''
        self.Yt_exp = np.transpose(np.zeros(len(self.b)))
        for i in range(self.N):
            self.Yt_exp[i] = self.Yt[i] * np.exp(-self.Lambda[i,i]*self.delta_t) + (1 - np.exp(-self.Lambda[i,i]*self.delta_t) ) *  \
                         self.b[i] 
        return self.V_inv.dot(self.Yt_exp)
