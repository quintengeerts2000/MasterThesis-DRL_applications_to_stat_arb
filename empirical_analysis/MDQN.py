import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
import math
#from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import os
import gym

device = device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

#torch.set_default_device(device)

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class DDQN_transf(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed, layer_type="ff"):
        super(DDQN_transf, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.ViT    = CNNTransformer() 
        self.head_1 = nn.Linear(8, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head_1, self.ff_1])
    
    def forward(self, input):
        """
        
        """
        x = self.ViT.extr_sig(input)
        x = torch.relu(self.head_1(x))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out
    
class DDQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed, layer_type="ff"):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head_1, self.ff_1])
    
    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class DDQN_alloc(nn.Module):
    def __init__(self, state_size, action_size,hidden_layers, seed, layer_type="ff"):
        super(DDQN_alloc, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(self.input_shape[0], hidden_layers[0]) # input = 30 -> 16
        self.ff_1 = nn.Linear(hidden_layers[0], hidden_layers[1])      # 16 -> 8
        self.ff_2 = nn.Linear(hidden_layers[1], hidden_layers[2])      # 8  -> 4

        self.ff_alloc = nn.Linear(hidden_layers[2] + 1, action_size)   # 4 features + current alloc -> actions
        weight_init([self.head_1, self.ff_1])
    
    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1, cont_action_space:bool=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.cont_act = cont_action_space 
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #print("before:", state,action,reward,next_state, done)
        n = state.shape[0]
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            #print("after:",state,action,reward,next_state, done)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    
    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        if self.cont_act:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        else:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class M_DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed,
                 use_transf=False,
                 add_alloc=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.ENTROPY_TAU = 0.03
        self.ALPHA       = 0.99
        self.LO          = -1
        self.Q_updates = 0

        self.action_step = 4
        self.last_action = None
    
        # Q-Network
        if not use_transf:
            self.qnetwork_local = DDQN(state_size, action_size,layer_size, seed).to(device)
            self.qnetwork_target = DDQN(state_size, action_size,layer_size, seed).to(device)
        else:
            self.qnetwork_local = DDQN_transf(state_size, action_size,layer_size, seed).to(device)
            self.qnetwork_target = DDQN_transf(state_size, action_size,layer_size, seed).to(device)
        
        if not add_alloc:
            self.qnetwork_local = DDQN(state_size, action_size,layer_size, seed).to(device)
            self.qnetwork_target = DDQN(state_size, action_size,layer_size, seed).to(device)
        else:
            self.qnetwork_local = DDQN_alloc(state_size, action_size,[16,8,4], seed).to(device)
            self.qnetwork_target = DDQN_alloc(state_size, action_size,[16,8,4], seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, 1)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):#, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 5000: #self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                #writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action 
                return action
            #self.action_step = 0
        else:
            self.action_step += 1
            return self.last_action
    
    def act_para(self, state, eps=0.):
        """Returns actions for given vector of states as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """
        
        if self.action_step == 4:
            state = np.array(state)
            n     = state.shape[0]
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
                if action_values.dim() == 3:
                    action = np.argmax(action_values.cpu().data.numpy(),axis=2)
                else:
                    action = np.argmax(action_values.cpu().data.numpy(),axis=1).reshape(1,-1)
                self.last_action = action
                return action
            else:
                action = np.random.choice(np.arange(self.action_size),(1,n))
                self.last_action = action 
                return action
            #self.action_step = 0
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()

        entropy_tau = self.ENTROPY_TAU
        alpha       = self.ALPHA
        lo          = self.LO

        states, actions, rewards, next_states, dones = experiences
        # Get predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach()
        # calculate entropy term with logsum 
        logsum = torch.logsumexp(\
                                (Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1))/entropy_tau , 1).unsqueeze(-1)

        tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - entropy_tau*logsum
        # target policy
        pi_target = F.softmax(Q_targets_next/entropy_tau, dim=1)
        Q_target = (self.GAMMA * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones)).sum(1)).unsqueeze(-1)
        
        # calculate munchausen addon with logsum trick
        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target)/entropy_tau, 1).unsqueeze(-1)
        log_pi = q_k_targets - v_k_target - entropy_tau*logsum
        munchausen_addon = log_pi.gather(1, actions)
        
        # calc munchausen reward:
        munchausen_reward = (rewards + alpha*torch.clamp(munchausen_addon, min=lo, max=0))
        
        # Compute Q targets for current states 
        Q_targets = munchausen_reward + Q_target
        
        # Get expected Q values from local model
        q_k = self.qnetwork_local(states)
        Q_expected = q_k.gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) #mse_loss
        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

def eval_runs(eps, frame):
    """
    Makes an evaluation run with the current epsilon
    """
    env = gym.make("CartPole-v0")
    reward_batch = []
    for i in range(5):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
        
    #writer.add_scalar("Reward", np.mean(reward_batch), frame)
    
def eval_runs(eps, frame):
    """
    Makes an evaluation run with the current epsilon
    """
    #env = gym.make("CartPole-v0")
    p = np.ones(N)
    #process = ornstein_uhlenbeck_process(theta,mu,sigma,delta_t)
    #N = 1
    process = OU_process_shuffler(N,T,L)
    r = 0.0
    tc = 0.0
    env = TradingEnvironment(process,T,r,p,mode='portfolio', max_pi=1, max_change=1,initial_wealth=0, transaction_costs=tc)

    reward_batch = []
    for i in range(5):
        state, _ = env.reset()
        state = dict_to_features(state)
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step([action_to_portfolio[action]])
            state = dict_to_features(state)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)


def dict_to_features(d):
    #return torch.FloatTensor(d['values']).unsqueeze(0).numpy()
    #return torch.FloatTensor([*d['values'], *d['portfolio'], d['wealth']])
    #return torch.FloatTensor([*d['values'], *d['portfolio']])
	return torch.FloatTensor([*d['values'], *d['mu'],*d['sigma'],*d['theta']])#,*d['alloc']])

def MDQN_train(env, timesteps):
    # PARAMETERS
    #seed = 100
    seed = np.random.randint(0,100000)

    #writer = SummaryWriter("runs/"+"DQN_LL_new_1")
    frames = timesteps
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 1e-2
    eps_frames=5000
    min_eps=0.025
    LR = 1e-3
    UPDATE_EVERY = 1
    n_step = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    eps_fixed = False
    SAVE_MODEL = False
    file_name = 'N1'

    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # action_size     = env.action_space.n #### going to fix this
    action_size       = 3
    #state_size        = env.observation_space.shape
    state_size        = [4]

    agent = M_DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        layer_size=64,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY, 
                        device=device, 
                        seed=seed)

    ##########################
    action_to_portfolio = {0:-1, 1:0, 2: 1}
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state, _ = env.reset()
    state = dict_to_features(state)
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        
        next_state, reward, done, _ = env.step([action_to_portfolio[action]])
        next_state = dict_to_features(next_state)
        agent.step(state, action, reward, next_state, done)#, writer)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
        
        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            #writer.add_scalar("Average100", np.mean(scores_window), frame)
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode +=1 
            state, _ = env.reset()
            state = dict_to_features(state)
            score = 0  
    return agent, output_history

class CNN_Block(nn.Module):
    def __init__(self, in_filters=1, out_filters=8, normalization=True, filter_size=2):
        super(CNN_Block, self).__init__()  
        self.in_filters = in_filters
        self.out_filters = out_filters
        
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((filter_size-1,0),0)
        
        self.normalization1 = nn.InstanceNorm1d(in_filters)
        self.normalization2 = nn.InstanceNorm1d(out_filters)
        self.normalization = normalization
       
    def forward(self, x): #x and out have dims (N,C,T) where C is the number of channels/filters
        if self.normalization:
            x = self.normalization1(x)
        out = self.left_zero_padding(x)
        out = self.conv1(out)
        out = self.relu(out)
        if self.normalization: 
            out = self.normalization2(out)
        out = self.left_zero_padding(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out + x.repeat(1,int(self.out_filters/self.in_filters),1)   
        return out
    
class CNNTransformer(nn.Module):
    def __init__(self, 
                 random_seed = 0, 
                 device = "cpu", # other options for device are e.g. "cuda:0"
                 normalization_conv = True, 
                 filter_numbers = [1,8], 
                 attention_heads = 4, 
                 use_convolution = True,
                 hidden_units = 2*8, 
                 hidden_units_factor = 2,
                 dropout = 0.25, 
                 filter_size = 2, 
                 use_transformer = True):
        
        super(CNNTransformer, self).__init__()
        if hidden_units and hidden_units_factor and hidden_units != hidden_units_factor * filter_numbers[-1]:
            raise Exception(f"`hidden_units` conflicts with `hidden_units_factor`; provide one or the other, but not both.")
        if hidden_units_factor:
            hidden_units = hidden_units_factor * filter_numbers[-1]
        self.random_seed = random_seed 
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        self.filter_numbers = filter_numbers
        self.use_transformer = use_transformer
        self.use_convolution = use_convolution and len(filter_numbers) > 0
        self.is_trainable = True
        
        self.convBlocks = nn.ModuleList()
        for i in range(len(filter_numbers)-1):
            self.convBlocks.append(
                CNN_Block(filter_numbers[i],filter_numbers[i+1],normalization=normalization_conv,filter_size=filter_size))
        self.encoder = nn.TransformerEncoderLayer(d_model=filter_numbers[-1], nhead=attention_heads, dim_feedforward=hidden_units, dropout=dropout)
        self.linear = nn.Linear(filter_numbers[-1],1)
        #self.softmax = nn.Sequential(nn.Linear(filter_numbers[-1],num_classes))#,nn.Softmax(dim=1))
                 
    def forward(self,x): #x has dimension (N,T)
        N,T = x.shape
        x = x.reshape((N,1,T))  #(N,1,T)
        if self.use_convolution:
            for i in range(len(self.filter_numbers)-1):
                x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
        x = x.permute(2,0,1)
        if self.use_transformer:
            x = self.encoder(x) #the input of the transformer is (T,N,C)
        return self.linear(x[-1,:,:]).squeeze() #this outputs the weights #self.softmax(x[-1,:,:]) #(N,num_classes)
    
    def extr_sig(self,x): #x has dimension (N,T)
        if len(x.shape) == 3:
            _,N,T = x.shape
        else:
            N,T = x.shape
        x = x.reshape((N,1,T))  #(N,1,T)
        if self.use_convolution:
            for i in range(len(self.filter_numbers)-1):
                x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
        x = x.permute(2,0,1)
        if self.use_transformer:
            x = self.encoder(x) #the input of the transformer is (T,N,C)
        return x[-1,:,:] 