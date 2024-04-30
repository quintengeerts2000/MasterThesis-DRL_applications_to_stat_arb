# public library imports
import pandas as pd
import os
import numpy as np
print("Current directory:", os.getcwd())
import sys
sys.path.append('.')
import torch
from collections import deque
from tqdm import tqdm
import os
import datetime as dt
import matplotlib.pyplot as plt
from IPython.display import clear_output

# own imports  
from MDQN import ReplayBuffer, M_DQN_Agent
from Trading_Simulator import TradingEnvironment
from Residual_Generator import PCA, IPCA, PCA_old
from Signal_Extractor import FourierExtractor, CNNTransformerExtractor, CumsumExtractor
from TD3 import ReplayBuffer_TD3, TD3
from QR_DQN import DQN_Agent

action_to_portfolio = {0:-1, 1:0, 2: 1}

# GLOBAL PARAMETERS

seed = 100  #np.random.randint(0,100000)

frames      = 10 #7000
BUFFER_SIZE = 1000000
BATCH_SIZE  = 128
GAMMA       = 0.99
TAU         = 1e-2
eps_frames  = 100
min_eps     = 0.001
LR          = 1e-3
UPDATE_EVERY= 5
n_step      = 1

MINIMUM_BUFFER  = 5000      # minimum amount of experiences before the DRL agent starts learning
EPISODE_LENGTH  = 25        # episode length, only used for telling when to show the performance
LOOKBACK_WINDOW = 252       # lookback window used for the PCA estimation
LOADING_WINDOW  = 60        # last 30 values are used to create the loadings for the factor model
SIGNAL_WINDOW   = 30        # the length of the window used for extracting the signal
WARMUP_TIME     = 0         # time needed to run the factor models before starting the strategy
RETRAIN_EVERY   = 250       # time before the feature extractor is retrained on new data

# PARAMETERS
tau  = 0.005 # Target network update rate

HIDDEN_DIM      = 64  #[64,64,64] #64
POLICY_NOISE    = 0.2 # Noise added to target policy during critic update
MAX_ACTION      = 1
POLICY_FREQ     = 2   # Frequency of delayed policy updates
EXPL_NOISE      = 0.1 # Std of Gaussian exploration noise
START_TIMESTEPS = 50  #25e3 # Time steps initial random policy is used
NOISE_CLIP      = 0.5 # Range to clip target policy noise
EVAL_FREQ       = 5e3 # How often (time steps) we evaluate

##########################
BATCH_SIZE_EXTR = 5

def M_DQN_test(price_df, facts, trans_cost):

    TRANS_COST = trans_cost[0]
    SHORT_COST = trans_cost[1]
    FACTORS    = facts

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    eps_fixed = False

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_size       = 3
    if TRANS_COST > 0:
        state_size        = [SIGNAL_WINDOW + 1]
        USE_ALLOC         = True
    else:
        state_size        = [SIGNAL_WINDOW]
        USE_ALLOC         = False

    agent = M_DQN_Agent(state_size=state_size,    
                            action_size=action_size,
                            layer_size=HIDDEN_DIM,
                            BATCH_SIZE=BATCH_SIZE, 
                            BUFFER_SIZE=BUFFER_SIZE, 
                            LR=LR, 
                            TAU=TAU, 
                            GAMMA=GAMMA, 
                            UPDATE_EVERY=UPDATE_EVERY, 
                            device=device, 
                            seed=seed,
                            add_alloc=False)

    ##########################

    residual_generator = PCA(price_data=price_df,
                            amount_of_factors = FACTORS,
                            loadings_window_size=LOADING_WINDOW,
                            lookback_window_size=LOOKBACK_WINDOW)

    env = TradingEnvironment(financial_dataset=price_df.dropna(axis=0, thresh=300),
                            residual_generator=residual_generator,
                            episode_length=EPISODE_LENGTH,
                            lookback_window=LOOKBACK_WINDOW,
                            loading_window=LOADING_WINDOW,
                            signal_window=SIGNAL_WINDOW,
                            transaction_costs=TRANS_COST,
                            short_cost=SHORT_COST,
                            add_alloc=USE_ALLOC)

    signal_extractor = FourierExtractor(signal_window=SIGNAL_WINDOW,
                                        add_alloc=USE_ALLOC)

    buffer           = ReplayBuffer(buffer_size=BUFFER_SIZE,
                                    batch_size=BATCH_SIZE,
                                    device=device,
                                    seed=seed,
                                    gamma=GAMMA,
                                    n_step=n_step)

    state, _ = env.warm_up(warmup_time=WARMUP_TIME)
    signal_extractor.train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date])
    features = state[env.tradeable_tickers].values.T # turns the dataframe for the requested tickers into a ndarray
    features = signal_extractor.extract(features)

    # INITIALISING SOME VARIABLES
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1

    #training loop
    for frame in range(1, frames+1):

        action = agent.act_para(features, eps)
        portfolio_allocations = np.vectorize(action_to_portfolio.get)(action) #actions need to be turned into allocations
        next_state, reward, done, _  = env.step(portfolio_allocations.T)
        
        # ADD TO BUFFER 
        for idx, ticker in enumerate(env.old_tradeable_tickers):
            if done[ticker].values[0]:
                # if the ticker is removed from the tradeable tickers, done = true and new_state is the last one
                buffer.add(state[ticker].values, action[0,idx], reward[ticker].values[0], state[ticker].values, True)
            else:
                # if the ticker is not removed from the tradeable tickers, done = false 
                buffer.add(state[ticker].values, action[0,idx], reward[ticker].values[0], next_state[ticker].values, False)

            # DEEP REINFORCEMENT LEARNING UPDATE
            if len(buffer) > MINIMUM_BUFFER and idx % UPDATE_EVERY == 0:
                experiences = buffer.sample()
                # extract the features using the current feature extractor
                states_buffer, actions_buffer, rewards_buffer, next_states_buffer, dones_buffer = experiences
                features_buffer      = torch.FloatTensor(signal_extractor.extract(states_buffer))
                next_features_buffer = torch.FloatTensor(signal_extractor.extract(next_states_buffer))
                experiences          = (features_buffer, actions_buffer, rewards_buffer, next_features_buffer, dones_buffer)
                # perform a learning stepre
                agent.learn(experiences)

        # FEATURE EXTRACTOR RETRAINING
        if frame % RETRAIN_EVERY == 0:
            signal_extractor.train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date]) 
        
        state     = next_state
        features  = next_state[env.tradeable_tickers].values.T
        features = signal_extractor.extract(features)

        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
    
    #save all the files now 
    env.evaluate_performance()
    env.total_pl.to_csv('results/results_pl_MDQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_rets.to_csv('results/res_rets_MDQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_alloc.to_csv('results/res_alloc_MDQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))

def TD3_test(price_df, facts, trans_cost):

    action_dim = 1
    max_action = 1

    TRANS_COST = trans_cost[0]
    SHORT_COST = trans_cost[1]
    FACTORS    = facts

    if TRANS_COST > 0:
        state_size        = [SIGNAL_WINDOW + 1]
        USE_ALLOC         = True
    else:
        state_size        = [SIGNAL_WINDOW]
        USE_ALLOC         = False

    kwargs = {
        "state_dim": state_size[0],
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": GAMMA,
        "tau": tau,
        'hidden_dimension': HIDDEN_DIM
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize policy
    kwargs["policy_noise"] = POLICY_NOISE * MAX_ACTION
    kwargs["noise_clip"]   = NOISE_CLIP * MAX_ACTION
    kwargs["policy_freq"]  = POLICY_FREQ
    kwargs["use_alloc"]    = False # USE_ALLOC
    policy = TD3(**kwargs)

    residual_generator = PCA(price_data=price_df,
                            amount_of_factors = FACTORS,
                            loadings_window_size=LOADING_WINDOW,
                            lookback_window_size=LOOKBACK_WINDOW)

    env = TradingEnvironment(financial_dataset=price_df.dropna(axis=0, thresh=300),
                            residual_generator=residual_generator,
                            episode_length=EPISODE_LENGTH,
                            lookback_window=LOOKBACK_WINDOW,
                            loading_window=LOADING_WINDOW,
                            signal_window=SIGNAL_WINDOW,
                            transaction_costs=TRANS_COST,
                            short_cost=SHORT_COST,
                            add_alloc=USE_ALLOC,
                            rebalance_every=1)

    buffer           = ReplayBuffer(buffer_size=BUFFER_SIZE,
                                    batch_size=BATCH_SIZE,
                                    device=device,
                                    seed=seed,
                                    gamma=GAMMA,
                                    n_step=n_step,
                                    cont_action_space=True)

    signal_extractor = FourierExtractor(add_alloc=USE_ALLOC, signal_window=SIGNAL_WINDOW)

    state, _ = env.warm_up(warmup_time=WARMUP_TIME)
    signal_extractor.train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date])
    features = state[env.tradeable_tickers].values.T # turns the dataframe for the requested tickers into a ndarray
    features = signal_extractor.extract(features)

    #training loop
    for frame in range(1, frames+1-WARMUP_TIME):
        # Select action randomly or according to policy
        if frame < START_TIMESTEPS:
            action = np.random.uniform(-1,1,(sum(env.tradeables),1))
        else:
            action = (
                policy.select_multiple_action(features)
                + np.random.normal(0, MAX_ACTION * EXPL_NOISE, size=(sum(env.tradeables),1))
            ).clip(-max_action, max_action)
        
        next_state, reward, done, info = env.step(action)
        
        # ADD TO BUFFER 
        for idx, ticker in enumerate(env.old_tradeable_tickers):
            if done[ticker].values[0]:
                # if the ticker is removed from the tradeable tickers, done = true and new_state is the last one
                buffer.add(state[ticker].values, action[idx,0], reward[ticker].values[0], state[ticker].values, False)
            else:
                # if the ticker is not removed from the tradeable tickers, done = false 
                buffer.add(state[ticker].values, action[idx,0], reward[ticker].values[0], next_state[ticker].values, True)

            # DEEP REINFORCEMENT LEARNING UPDATE
            if len(buffer) > MINIMUM_BUFFER and idx % UPDATE_EVERY == 0:
                experiences = buffer.sample()
                # extract the features using the current feature extractor
                states_buffer, actions_buffer, rewards_buffer, next_states_buffer, dones_buffer = experiences
                features_buffer      = torch.FloatTensor(signal_extractor.extract(states_buffer))
                next_features_buffer = torch.FloatTensor(signal_extractor.extract(next_states_buffer))
                experiences          = (features_buffer, actions_buffer, rewards_buffer, next_features_buffer, dones_buffer)
                # perform a learning step
                policy.train(experiences)

        # FEATURE EXTRACTOR RETRAINING
        if frame % RETRAIN_EVERY == 0:
            signal_extractor.re_train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date],
                                    sample_size = BATCH_SIZE_EXTR) 
        
        state     = next_state
        features  = next_state[env.tradeable_tickers].values.T
        features  = signal_extractor.extract(features)
    #save all the files now 
    env.evaluate_performance()
    env.total_pl.to_csv('results/results_pl_TD3_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_rets.to_csv('results/res_rets_TD3_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_alloc.to_csv('results/res_alloc_TD3_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))

def QR_DQN_test(price_df, facts, trans_cost):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    eps_fixed = False

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_size       = 3

    TRANS_COST = trans_cost[0]
    SHORT_COST = trans_cost[1]
    FACTORS    = facts

    if TRANS_COST > 0:
        state_size        = [SIGNAL_WINDOW + 1]
        USE_ALLOC         = True
    else:
        state_size        = [SIGNAL_WINDOW]
        USE_ALLOC         = False

    agent = DQN_Agent(state_size=state_size,    
                            action_size=action_size,
                            layer_size=HIDDEN_DIM,
                            Network = 'DDQN',
                            n_step = n_step,
                            BATCH_SIZE=BATCH_SIZE, 
                            BUFFER_SIZE=BUFFER_SIZE, 
                            LR=LR, 
                            TAU=TAU, 
                            GAMMA=GAMMA, 
                            UPDATE_EVERY=UPDATE_EVERY, 
                            device=device, 
                            seed=seed)

    residual_generator = PCA(price_data=price_df,
                            amount_of_factors = FACTORS,
                            loadings_window_size=LOADING_WINDOW,
                            lookback_window_size=LOOKBACK_WINDOW)

    env = TradingEnvironment(financial_dataset=price_df.dropna(axis=0, thresh=300),
                            residual_generator=residual_generator,
                            episode_length=EPISODE_LENGTH,
                            lookback_window=LOOKBACK_WINDOW,
                            loading_window=LOADING_WINDOW,
                            signal_window=SIGNAL_WINDOW,
                            transaction_costs=TRANS_COST,
                            short_cost=SHORT_COST,
                            add_alloc=USE_ALLOC)

    signal_extractor = FourierExtractor(signal_window=SIGNAL_WINDOW,
                                        add_alloc=USE_ALLOC)

    buffer           = ReplayBuffer(buffer_size=BUFFER_SIZE,
                                    batch_size=BATCH_SIZE,
                                    device=device,
                                    seed=seed,
                                    gamma=GAMMA,
                                    n_step=1)

    state, _ = env.warm_up(warmup_time=WARMUP_TIME)
    signal_extractor.train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date])
    features = state[env.tradeable_tickers].values.T # turns the dataframe for the requested tickers into a ndarray
    features = signal_extractor.extract(features)

    # INITIALISING SOME VARIABLES
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1

    #training loop
    for frame in range(1, frames+1):

        action = agent.act_para(features, eps)
        portfolio_allocations = np.vectorize(action_to_portfolio.get)(action) #actions need to be turned into allocations
        next_state, reward, done, info = env.step(portfolio_allocations.T)
        
        # ADD TO BUFFER 
        for idx, ticker in enumerate(env.old_tradeable_tickers):
            if done[ticker].values[0]:
                # if the ticker is removed from the tradeable tickers, done = true and new_state is the last one
                buffer.add(state[ticker].values, action[0,idx], reward[ticker].values[0], state[ticker].values, True)
            else:
                # if the ticker is not removed from the tradeable tickers, done = false 
                buffer.add(state[ticker].values, action[0,idx], reward[ticker].values[0], next_state[ticker].values, False)

            # DEEP REINFORCEMENT LEARNING UPDATE
            if len(buffer) > MINIMUM_BUFFER and idx % UPDATE_EVERY == 0:
                experiences = buffer.sample()
                # extract the features using the current feature extractor
                states_buffer, actions_buffer, rewards_buffer, next_states_buffer, dones_buffer = experiences
                features_buffer      = torch.FloatTensor(signal_extractor.extract(states_buffer))
                next_features_buffer = torch.FloatTensor(signal_extractor.extract(next_states_buffer))
                experiences          = (features_buffer, actions_buffer, rewards_buffer, next_features_buffer, dones_buffer)
                # perform a learning stepre
                agent.learn(experiences)

        # FEATURE EXTRACTOR RETRAINING
        if frame % RETRAIN_EVERY == 0:
            signal_extractor.train(train_data = env.res_rets.loc[env.date - dt.timedelta(days=WARMUP_TIME):env.date]) 
        
        state     = next_state
        features  = next_state[env.tradeable_tickers].values.T
        features = signal_extractor.extract(features)

        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
    #save all the files now
    env.evaluate_performance()
    env.total_pl.to_csv('results/results_pl_QR_DQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_rets.to_csv('results/res_rets_QR_DQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))
    env.res_alloc.to_csv('results/res_alloc_QR_DQN_tc-{}_sc-{}_fcts-{}.csv'.format(TRANS_COST, SHORT_COST, FACTORS))

    

if __name__ == '__main__':
    transaction_costs = [(0.0,0.0), (0.0001, 0.00005), (0.0005, 0.0001)]
    factors           = [0, 1, 3, 5, 10, 15]
    # USE THE DATASET FROM WHARTON RESEARCH DATA SERVICES
    filename = 'wrds_daily_returns.csv'
    price_df = pd.read_csv(filename)
    price_df.set_index('date', inplace=True)
    price_df.index = pd.to_datetime(price_df.index)

    price_df = price_df.loc[:dt.datetime(2022,12,31)]
    price_df = price_df.replace(['C','B'],np.nan).astype(float)
    print('starting...')
    total = len(transaction_costs) * len(factors)
    count = 1
    for f in factors:
        for tc in transaction_costs:
            print('M_DQN test: test {} of {}'.format(count,total))
            M_DQN_test(price_df=price_df,
                       facts=f,
                       trans_cost=tc)
            print('TD3 test: test {} of {}'.format(count,total))
            TD3_test(price_df=price_df,
                     facts=f,
                     trans_cost=tc)
            print('QR_DQN test: test {} of {}'.format(count,total))
            QR_DQN_test(price_df=price_df,
                       facts=f,
                       trans_cost=tc)
            count += 1
