import numpy as np
from scipy.linalg import eig, inv,cholesky
N = 100

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

# Before starting the environment needs to be made

class TradingEnvironment():
    def __init__(self, process, T, r, p):
        self.process = process
        self.T   = T
        self.L   = int(T/process.delta_t)
        self.N   = process.N
        self.idx = 0
        self.r   = r
        self.p   = p
        
        # setup the environment
        self.process.reset()
        self.X   = np.zeros((self.N,self.L))
        self.W   = np.zeros(self.L)

    def train(self):
        #TODO: train and eval need to be used in the future for more complex reward functions
        self.train = True
    
    def eval(self):
        self.train = False
        
    def reset(self):
        # reset the environment
        self.process.reset()
        self.X    = np.zeros((self.N,self.L))
        self.idx  = 0
        self.W    = np.zeros(self.L)
        state     = np.zeros(2*self.N + 1)
        state[:N] = self.process.X0.reshape((1,-1))
        state[-1] = self.T - self.process.t
        return state

    def step(self, pi):
        self.idx += 1
        X_t_exp   = self.process.expected_val()
        X_t       = self.process.step().reshape((-1,1))
        self.t    = self.process.t
        
        self.X[:,self.idx] = X_t.reshape((1,-1))

        dW_t = pi.squeeze().dot(self.X[:,self.idx] - self.X[:,self.idx-1]) + (self.W[self.idx-1] - pi.squeeze().dot(self.p))*self.r * self.process.delta_t
        self.W[self.idx] = self.W[self.idx-1] + dW_t


        state         = np.zeros(2*self.N + 1)
        state[:N]     = X_t.reshape((1,-1))  # first N values are the process values themselves at timestep t 
        state[N:-1]   = pi   # next N values are the previous pi 
        #state[-1]     = self.T - self.t # final state variable is the time left in the episode
        state[-1]     = 0
        if self.train:
            reward = pi.squeeze().dot(X_t_exp - self.X[:,self.idx-1]) + (self.W[self.idx-1] - pi.squeeze().dot(self.p))*self.r * self.process.delta_t
        else:
            reward = dW_t
        #done = (T - self.process.delta_t <= self.t)
        done = (self.L-1 == self.idx)
        return state, reward.item(), done, {}