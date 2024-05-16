import gym
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import time
# from Residual_Generator import PCA_old #TODO: remove


class TradingEnvironment(gym.Env):
    def __init__(self, 
                 financial_dataset:pd.DataFrame,
                 residual_generator,
                 episode_length:int=100,
                 lookback_window:int=252,
                 loading_window:int=60,
                 signal_window:int=60,
                 transaction_costs:float=0.0,
                 short_cost:float=0.0,
                 rebalance_every:int=1, 
                 add_alloc=False,
                 use_log_returns=False) -> None:
        '''
        financial_dataset: is a dataframe containing the adjusted prices of all the assets
        residual_generator: a function, called every timestep, whose input is a lookback window of asset prices, and then 
                            calculates the residual portfolios at that time.w
        '''
        super().__init__()

        self.L, self.N = financial_dataset.shape # amount of datapoints, amount of stocks
        financial_dataset = financial_dataset.astype(float)
        self.data      = financial_dataset       # dataset 
        self.returns   = financial_dataset #financial_dataset.pct_change(1,fill_method=None) # compute returns 
        self.res_rets  = pd.DataFrame(index=self.data.index, columns=self.data.columns) # dataframe used to store the residual returns
        self.res_alloc = pd.DataFrame(index=self.data.index, columns=self.data.columns).replace(np.nan, 0) # dataframe used to store the chosen allocation per residual portfolio
        self.asset_alloc= pd.DataFrame(index=self.data.index, columns=self.data.columns) # dataframe used to store the allocation in the original asset space
        self.total_pl  = pd.DataFrame(index=self.data.index, columns=['strategy'])

        # both generic functions so they can be swapped in the future
        self.res_gen   = residual_generator 

        self.ep_N      = episode_length    # amount of timesteps until an 'episode' is over
        self.tc        = transaction_costs # transaction cost used
        self.sc        = short_cost        # cost to keep a short position
        self.lbw       = lookback_window   # lookback window used for pca 
        self.sig_win   = signal_window     # lookback window used for signals
        self.load_win  = loading_window
        self.rebalance_every = rebalance_every
        self.add_alloc = add_alloc
        self.use_log_ret = use_log_returns

        self.t         = self.lbw + 1 #current timestep idx position in the large dataset
        self.ep        = 0  # current episode
        self.max_ep    = (self.L - self.lbw - self.sig_win) // self.ep_N # maximal amount of episodes possible with the data

        self.t_ep    = 0 # current timestep in the epsisode (max is self.ep_N)

    def warm_up(self, warmup_time:int=0):
        # if at initialisation
        assert self.ep == 0

        self.res_portf     = np.zeros((self.N,self.N))
        self.active_stocks = np.zeros((self.N,),dtype=bool)
        
        for _ in range(self.sig_win):
            # start by computing the residual returns
            self.iter_step()
            self.res_rets_step()

            # calculate the new residual portfolio weights at time t (in pandas :t+1, means the last row is at time t)
            if self.t % self.rebalance_every == 0:
                # self.res_portf, self.active_stocks  = PCA_old(self.data.iloc[self.t - self.lbw: self.t+1],
                #                                             amount_of_factors=5,
                #                                             loadings_window_size=self.load_win)

                self.res_portf, self.active_stocks  = self.res_gen.step(self.date)
        
        # initialise the allocation vectors 
        self.old_alloc       = np.zeros((self.N,self.N))
        self.old_alloc_total = np.zeros((self.N,))

        observation = self._get_next_obs()
        while (sum(self.tradeables)) == 0 or (self.t < warmup_time):
            n = sum(self.tradeables)
            observation, _, _, _ = self.step(np.zeros((n,1)))
        self.start_t = self.t
        info = {'used_stocks': self.active_stocks}
        print('environment ready')
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

    @property
    def date(self):
        return self.data.index[self.t]
    
    @property
    def tradeable_tickers(self):
        return self.data.columns[self.tradeables]
    
    @property
    def old_tradeable_tickers(self):
        return self.data.columns[self.old_tradeables]

    def _get_next_obs(self):
        # the the returns at time (t) from the residual portfolios generated at time (t-1)
        # generate the signal vector from the residual portfolio returns 
        # generate the new residual portfolios to trade in
        observation = self.res_rets.iloc[self.t - self.sig_win + 1: self.t + 1]
        self.tradeables = ~np.any(np.isnan(observation.values.astype(float)), axis = 0).ravel()
        if self.add_alloc:
            # add previous allocation to the observation
            observation = pd.DataFrame(columns=self.data.columns, data=np.vstack([observation.values, self.res_alloc.iloc[self.t].values]))    
        return observation.astype(float)

    def calculate_transaction_cost(self):
        '''
        This calculation is based on the market friction model from Boyd et al. (2017)

        tc_vector contains the individual transaction costs incurred by trading in each of the portfolios
        tc_total is a real that contains the true total transaction cost incurred by the 
        '''
        tc_vector = self.tc * np.linalg.norm(self.new_alloc - self.old_alloc,1,axis=1) \
                        + self.sc * np.linalg.norm(np.minimum(self.new_alloc, np.zeros((self.N,self.N))),1,axis=1)

        tc_total = self.tc * np.linalg.norm(self.new_alloc_total - self.old_alloc_total,1) \
                        + self.sc * np.linalg.norm(np.minimum(self.new_alloc_total, np.zeros(self.N)),1)
        
        return tc_vector, tc_total

    def evaluate_performance(self):
        perf = self.total_pl.iloc[self.t - self.ep_N + 1:self.t + 1].values
        # Compute cumulative return
        if self.use_log_ret:
            cumulative_return = np.exp(np.sum(perf)) - 1
        else:
            cumulative_return = np.prod(1 + perf) - 1
        # Calculate annualized return
        annualized_return = (1 + cumulative_return)**(252/self.ep_N) - 1
        # Calculate volatility (standard deviation)
        volatility = np.std(perf)
        # Annualize volatility
        annualized_volatility = volatility * np.sqrt(252)
        print("\r Episode: {} -- Annualized Return: {}% -- Annualized Volatility: {}% -- Annualized Sharpe: {}"
              .format(self.ep, round(annualized_return*100,2),
                      round(annualized_volatility*100,2),
                round(annualized_return / annualized_volatility, 2)))
        
        perf_tot = self.total_pl.iloc[self.start_t + 1:self.t + 1].values
        # Compute cumulative return
        if self.use_log_ret:
            cumulative_return = np.exp(np.sum(perf_tot)) - 1
        else:
            cumulative_return = np.prod(1 + perf_tot) - 1
        # Calculate annualized return
        annualized_return = (1 + cumulative_return)**(252/(self.t - self.start_t)) - 1
        # Calculate volatility (standard deviation)
        volatility = np.std(perf_tot)
        # Annualize volatility
        annualized_volatility = volatility * np.sqrt(252)
        print("\r Episode: {} -- Total ann. Return: {}% -- Total ann. Volatility: {}% -- Total ann. Sharpe: {}"
              .format(self.ep, round(annualized_return*100,2),
                      round(annualized_volatility*100,2),
                round(annualized_return / annualized_volatility, 2)))
        self.ep += 1
    
    def step(self,action):
        '''
        Action is a vector of size N_t, the size of which can vary according to the active stocks at the time
        it represents the chosen loadings in the residual portfolios. 

        The reward will be a vector of the rewards generated by the individual portfolios and their allocation, and 
        will be be returned to the reinforcement learning agent as a reward signal.
        However, in the real environment there is a shared portfolio that will also be updated.
        '''
        assert len(action) == sum(self.tradeables), 'wrong size of action/allocation vector supplied'

        ########### Time: t-1 #############

        # "invest" the chosen amount Action
        allocation_in_residuals = np.zeros((self.N,1))
        allocation_in_residuals[self.tradeables] = action

        # calculate the new allocation in terms of the true asset space
        self.new_alloc = (self.res_portf * np.tile(allocation_in_residuals, (1, self.N)))                 # per residual portf.
        self.new_alloc_total  = self.new_alloc.sum(axis=0) 
        self.new_alloc_total /= (np.linalg.norm(self.new_alloc_total,1)+ 1e-8)  # for the entire (normalized) portf.

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
        reward        = pd.DataFrame(data=reward.reshape(1,-1), columns=self.data.columns)

        # calculate the general profit made by the whole portfolio
        change_total  = self.new_alloc_total @ self.returns.iloc[self.t].replace(np.nan,0).values
        self.p_l      = change_total - self.tc_total
        self.total_pl.loc[self.date, 'strategy']  = self.p_l

        # reward[self.tradeable_tickers] = self.p_l
        
        # save the old allocations and store them 
        self.old_alloc       = self.new_alloc.copy()
        self.old_alloc_total = self.new_alloc_total.copy()
        self.res_alloc.loc[self.date] = allocation_in_residuals.flatten()
        self.asset_alloc.loc[self.date] = self.new_alloc_total.flatten()

        # prepare the next observation, the new returns were already calulated in res_rets_step
        self.old_tradeables = self.tradeables
        observation = self._get_next_obs()

        # calculate the new residual portfolio weights at time t
        if self.t % self.rebalance_every == 0:
            # self.res_portf, self.active_stocks   = PCA_old(self.data.iloc[self.t - self.lbw: self.t+1],
            #                                             amount_of_factors=5,
            #                                             loadings_window_size=self.load_win)
            self.res_portf, self.active_stocks   = self.res_gen.step(self.date)

        # keep track of which stocks are added and removed
        changes = self.tradeables.astype(int) - self.old_tradeables.astype(int)
        added   = self.data.columns[changes == 1]
        removed = self.data.columns[changes == -1]
        info = {'added_tickers':added, 'removed_tickers':removed} # give the info of which stocks were added/removed 
        done = pd.DataFrame(data=(changes == -1).reshape(1,-1), columns=self.data.columns)

        return observation, reward, done, info
