import gym
import pandas as pd
import datetime as dt
import numpy as np
from numpy.linalg import eig, norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time


class TradingEnvironment(gym.Env):
    def __init__(self, 
                 financial_dataset:pd.DataFrame,
                 residual_generator,
                 signal_generator,
                 episode_length:int=100,
                 lookback_window:int=252,
                 signal_window:int=60,
                 transaction_costs:float=0.0,
                 short_cost:float=0.0) -> None:
        '''
        financial_dataset: is a dataframe containing the adjusted prices of all the assets
        residual_generator: a function, called every timestep, whose input is a lookback window of asset prices, and then 
                            calculates the residual portfolios at that time.
        signal_generator:  a function, called every timestep, whose input is a lookback window of residual returns, that extracts 
                            the signal from the lookback window
        '''
        super().__init__()

        self.L, self.N = financial_dataset.shape # amount of datapoints, amount of stocks
        self.data      = financial_dataset  # dataset 
        self.returns   = financial_dataset.pct_change(1,fill_method=None) # compute returns 
        self.res_rets  = pd.DataFrame(index=self.data.index, columns=self.data.columns) # dataframe used to store the residual returns
        self.res_alloc = pd.DataFrame(index=self.data.index, columns=self.data.columns) # dataframe used to store the chosen allocation per residual portfolio
        self.asset_alloc= pd.DataFrame(index=self.data.index, columns=self.data.columns) # dataframe used to store the allocation in the original asset space
        self.total_pl  = pd.DataFrame(index=self.data.index, columns=['strategy'])

        # both generic functions so they can be swapped in the future
        self.res_gen   = residual_generator 
        self.sig_gen   = signal_generator

        self.ep_N      = episode_length    # amount of timesteps until an 'episode' is over
        self.tc        = transaction_costs # transaction cost used
        self.sc        = short_cost        # cost to keep a short position
        self.lbw       = lookback_window   # lookback window used for pca 
        self.sig_win   = signal_window     # lookback window used for signals

        self.t         = self.lbw + 1 #current timestep idx position in the large dataset
        self.ep        = 0  # current episode
        self.max_ep    = (self.L - self.lbw - self.sig_win) // self.ep_N # maximal amount of episodes possible with the data

        self.t_ep    = 0 # current timestep in the epsisode (max is self.ep_N)

    def warm_up(self):
        # if at initialisation
        assert self.ep == 0

        self.res_portf     = np.zeros((self.N,self.N))
        self.active_stocks = np.zeros((self.N,),dtype=bool)
        
        for _ in range(self.sig_win):
            # start by computing the residual returns
            self.iter_step()
            self.res_rets_step()

            # calculate the new residual portfolio weights at time t (in pandas :t+1, means the last row is at time t)
            self.res_portf, self.active_stocks  = self.res_gen(self.data.iloc[self.t - self.lbw: self.t+1],
                                                        amount_of_factors=5,
                                                        loadings_window_size=self.sig_win)
        
        # initialise the allocation vectors 
        self.old_alloc       = np.zeros((self.N,self.N))
        self.old_alloc_total = np.zeros((self.N,))

        observation = self._get_next_obs()
        while sum(self.tradeable_stocks) == 0:
            n = sum(self.tradeable_stocks)
            observation, _, _, _ = self.step(np.zeros((n,1)))
        info = {'used_stocks': self.active_stocks}
        return observation, info
    
    def res_rets_step(self):
        self.res_portf_previous = self.res_portf.copy()
        self.res_rets.iloc[self.t,self.active_stocks] = (self.res_portf @ \
                        self.returns.iloc[self.t].replace(np.nan,0).values)[self.active_stocks]
        
    def iter_step(self):
        '''
        keeps track of all timesteps
        '''
        self.t    += 1
        self.t_ep += 1
        if self.t_ep >= self.ep_N:
            self.ep += 1

    @property
    def date(self):
        return self.data.index[self.t]
    
    @property
    def trade_able_ticker(self):
        return self.data.columns[self.tradeable_stocks]

    def _get_next_obs(self):
        # the the returns at time (t) from the residual portfolios generated at time (t-1)
        # generate the signal vector from the residual portfolio returns 
        # generate the new residual portfolios to trade in
        self.tradeable_stocks = ~np.any(np.isnan(self.res_rets.iloc[self.t - self.sig_win + 1: self.t + 1].values.astype(float)), axis = 0).ravel()
        obs = self.sig_gen(self.res_rets.iloc[self.t - self.sig_win + 1: self.t + 1, self.tradeable_stocks])
        return obs

    def calculate_transaction_cost(self):
        '''
        This calculation is based on the market friction model from Boyd et al. (2017)

        tc_vector contains the individual transaction costs incurred by trading in each of the portfolios
        tc_total is a real that contains the true total transaction cost incurred by the 
        '''
        tc_vector = self.tc * np.linalg.norm(self.new_alloc - self.old_alloc,1,axis=1) \
                        + self.sc * np.linalg.norm(np.minimum(self.new_alloc, np.zeros((self.N,self.N))))

        tc_total = self.tc * np.linalg.norm(self.new_alloc_total -self.old_alloc_total,1) \
                        + self.sc * np.linalg.norm(np.minimum(self.new_alloc_total, np.zeros(self.N)))
        return tc_vector, tc_total

    def evaluate_performance(self):
        perf = self.total_pl.iloc[self.t - self.ep_N].values

        # Compute cumulative return
        cumulative_return = np.prod(1 + perf) - 1

        # Calculate annualized return
        annualized_return = (1 + cumulative_return)**(252/self.ep_N) - 1

        # Calculate volatility (standard deviation)
        volatility = np.std(perf)

        # Annualize volatility
        annualized_volatility = volatility * np.sqrt(252)

        print("Annualized Return:", annualized_return)
        print("Annualized Sharpe (Shape):", annualized_return / annualized_volatility)

    
    def step(self,action):
        '''
        Action is a vector of size N_t, the size of which can vary according to the active stocks at the time
        it represents the chosen loadings in the residual portfolios. 

        The reward will be a vector of the rewards generated by the individual portfolios and their allocation, and 
        will be be returned to the reinforcement learning agent as a reward signal.
        However, in the real environment there is a shared portfolio that will also be updated.
        '''
        assert len(action) == sum(self.tradeable_stocks), 'wrong size of action/allocation vector supplied'

        ########### Time: t-1 #############

        # "invest" the chosen amount Action
        allocation_in_residuals = np.zeros((self.N,1))
        allocation_in_residuals[self.tradeable_stocks] = action

        # calculate the new allocation in terms of the true asset space
        self.new_alloc = (self.res_portf * np.tile(allocation_in_residuals, (1, self.N)))                 # per residual portf.
        self.new_alloc_total  = self.new_alloc.sum(axis=0) 
        self.new_alloc_total /= np.linalg.norm(self.new_alloc_total,1)  # for the entire (normalized) portf.

        # we incur a transaction cost through this new allocation
        self.tc_vector, self.tc_total = self.calculate_transaction_cost()

        ########### Time: t  #############
        # then we need to step
        self.iter_step()
        self.res_rets_step()
        
        # now calculate the effect of the allocation for each individual stock 
        #TODO: debug this!!
        portf_change  = self.new_alloc @ self.returns.iloc[self.t].replace(np.nan,0).values
        reward        = portf_change - self.tc_vector

        # calculate the general profit made by the whole portfolio
        change_total  = self.new_alloc_total @ self.returns.iloc[self.t].replace(np.nan,0).values
        p_l           = change_total - self.tc_total
        self.total_pl.loc[self.date, 'strategy']  = p_l

        # save the old allocations and store them 
        self.old_alloc       = self.new_alloc.copy()
        self.old_alloc_total = self.new_alloc_total.copy()
        self.res_alloc.loc[self.date] = allocation_in_residuals.flatten()
        self.asset_alloc.loc[self.date] = self.new_alloc_total.flatten()

        # prepare the next observation, the new returns were already calulated in res_rets_step
        observation = self._get_next_obs()

        # calculate the new residual portfolio weights at time t
        self.res_portf, self.active_stocks   = self.res_gen(self.data.iloc[self.t - self.lbw: self.t+1],
                                                    amount_of_factors=5,
                                                    loadings_window_size=self.sig_win)

        if self.t % self.ep_N == 0:
            done = True
        else:
            done = False
        info = {}
        return observation, reward, done, info

def fourier_signal_extractor(residuals_data:pd.DataFrame, output_size:int=30):
    '''
    All the data input in this function should be considered in sample
    '''
    L, N = residuals_data.shape
    assert L >= output_size, "can't calculate fourier transform for more than the input amount of data"
    res_window = (residuals_data + 1).cumprod().values[1:,:] - 1
    Fourier    = np.fft.rfft(res_window,axis=0, n=output_size//2)
    n_f        = Fourier.shape[0]
    out        = np.zeros((N,n_f*2))
    out[:,:n_f]= np.real(Fourier).T
    out[:,n_f:]= np.imag(Fourier).T #geen idee maar in andere papers doen ze dit ook
    return out

def pca_res_gen(price_data:pd.DataFrame, 
                amount_of_factors:int=5,
                loadings_window_size:int=60)-> np.ndarray:
    '''
    Calculates the pca portfolio given a dataset with prices
    '''

    T, N         = price_data.shape 
    assert loadings_window_size < T, 'loading window larger than length of dataset supplied' 

    rets         = price_data.pct_change(1,fill_method=None).iloc[1:].to_numpy()
    idxsSelected = ~np.any(np.isnan(rets), axis = 0).ravel()
    if idxsSelected.sum() == 0:
            return np.zeros((N,N))
    
    rets_is     = rets[:,idxsSelected] # in sample returns: used for generating the portfolio

    # Calculate PCA
    rets_mean       = np.mean(rets_is, axis=0,keepdims=True)
    rets_vol        = np.sqrt(np.mean((rets_is-rets_mean)**2,axis=0,keepdims=True))
    rets_normalized = (rets_is - rets_mean) / rets_vol
    Corr            = np.dot(rets_normalized.T, rets_normalized)
    _, eigenVectors = np.linalg.eigh(Corr)

    # Calculate loadings
    w           = eigenVectors[:,-amount_of_factors:].real  
    R           = rets_is[-loadings_window_size:,:]
    wtR         = R @ w  
    regr        = LinearRegression(fit_intercept=False, n_jobs=-1).fit(wtR,R)
    beta        = regr.coef_                                                    #beta
    psi         = (np.eye(beta.shape[0]) - beta @ w.T)

    # Calculate residual returns
    residual_portf = np.zeros((N,N))
    i = 0
    for idx, val in enumerate(idxsSelected):
         if val:
               residual_portf[idx,idxsSelected] = psi[i,:].reshape([1,-1])
               i += 1
    return residual_portf, idxsSelected