import numpy as np
from scipy.linalg import eig, inv

import gym
from gym import spaces

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
    
    def reset(self, X0=None):
        #initialise the values at t=0
        if X0 is None:
            self.X0 = np.zeros(self.N)
            self.Yt = np.transpose(np.zeros(len(self.b)))
        else:
            self.Yt = self.V.dot(X0)
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

class TradingEnvironment(gym.Env):
    def __init__(self, process, T, r, p, max_pi=np.inf, max_change=np.inf, initial_wealth=100, mode='intensity'):

        self.process = process
        self.T   = T
        self.L   = int(T/process.delta_t)
        self.N   = process.N
        self.idx = 0
        self.r   = r
        self.p   = p
        self.alloc = list()
        self.W0  = initial_wealth
        self.mode = mode

        self.max_pi = max_pi
        self.max_change = max_change

        self.observation_space = spaces.Dict({
            "values": spaces.Box(low=-np.inf, high= np.inf, shape=(self.N,),dtype=np.float32),
            "portfolio": spaces.Box(shape=(self.N,),low=-max_pi,high=max_pi,dtype=np.float32),
            "wealth": spaces.Box(low= -np.inf, high=np.inf, shape=())
        })

        self.action_space = spaces.Box(low=-max_change,high=max_change, shape=(self.N,))
        
        # setup the environment
        self.X   = np.zeros((self.N,self.L))
        self.W   = np.zeros(self.L)
    
    def _get_obs(self):
        return {"values":self.X_t, "portfolio": self.pi_t, "wealth": self.W_t}
    
    def _get_info(self):
        return {"expected_values": self.exp_val}

        
    def reset(self, seed=None):
        # reset the environment
        super().reset(seed=seed)
        X0 = self.np_random.multivariate_normal(self.process.mu.flatten(), self.process.Var) #TODO: initialisatie niet helemaal kosher
        self.process.reset(X0=X0)
    
        self.X      = np.zeros((self.N,self.L+1))
        self.X[:,0] = X0
        self.W     = np.zeros(self.L+1)
        self.W[0]  = self.W0
        self.alloc = list()
        self.idx    = 0

        self.pi_t   = np.zeros(self.N)
        self.X_t    = X0
        self.W_t    = self.W0
        self.exp_val= self.process.mu

        observation = self._get_obs()
        info        = self._get_info()
        return observation, info

    def step(self, action):
        # set portfolio for time t
        if self.mode == 'intensity':
            self.pi_t  = np.clip(self.pi_t + action,-self.max_pi,self.max_pi).flatten()
        elif self.mode == 'portfolio':
            self.pi_t  = np.clip(action, -self.max_pi, self.max_pi).flatten()
        else:
            raise ValueError('{} is not a valid mode for runnning the trading environment'.format(self.mode))
        # perform a step in the environment t -> t+1
        self.idx += 1
        self.X_t       = self.process.step()
        self.t         = self.process.t

        # calculate the return from portfolio of time t with returns of time t+1
        dW_t = self.pi_t.squeeze().dot(self.X_t.flatten() - self.X[:,self.idx-1]) #+ (self.W[self.idx-1] - abs(self.pi_t.squeeze().dot(self.p)))*self.r * self.process.delta_t
        self.W_t = self.W[self.idx-1] + dW_t
        self.exp_val   = self.process.expected_val()

        # save the values 
        self.X[:,self.idx] = self.X_t
        self.W[self.idx] = self.W_t 
        self.alloc.append(self.pi_t)

        observation = self._get_obs()
        reward = dW_t.item()
        done = (self.L-1 == self.idx)
        info        = self._get_info()
        return observation, reward, done, info