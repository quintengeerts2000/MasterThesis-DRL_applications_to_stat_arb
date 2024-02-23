import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
import math
#from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym

device = device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

#torch.set_default_device(device)

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class QR_DQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, n_step, seed, N, layer_type="ff"):
        super(QR_DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N

        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size*N)
        weight_init([self.head_1, self.ff_1])


    
    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out.view(input.shape[0], self.N, self.action_size)

    def get_action(self,input, risk_aversion=0):
        x = self.forward(input)
        return x.mean(dim=1) - risk_aversion * x.var(dim=1)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
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
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #print("before:", state,action,reward,next_state, done)
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 Network,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
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
        self.Q_updates = 0
        self.n_step = n_step
        self.N = 8 #32
        self.quantile_tau = torch.FloatTensor([i/self.N for i in range(1,self.N+1)]).to(device)

        self.action_step = 4
        self.last_action = None

        # Q-Network
        self.qnetwork_local = QR_DQN(state_size, action_size,layer_size, n_step, seed, self.N).to(device)
        self.qnetwork_target = QR_DQN(state_size, action_size,layer_size, n_step, seed, self.N).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):#, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 5000:#self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                #writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.,risk_aversion=0.):
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
                action_values = self.qnetwork_local.get_action(state,risk_aversion=risk_aversion)
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

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().cpu() #.max(2)[0].unsqueeze(1) #(batch_size, 1, N)

        #TODO: hier aanpassen voor risk aware te maken
        action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True) #action indx is here to maximise avg Q
        #action_indx = torch.argmax(Q_targets_next.mean(dim=1) / Q_targets_next.var(dim=1), dim=1, keepdim=True) #action indx is here to maximise avg Q
        #action_indx = torch.argmax(Q_targets_next.mean(dim=1) - 2* Q_targets_next.var(dim=1), dim=1, keepdim=True) #action indx is here to maximise avg Q
        Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,2)

        assert Q_targets_next.shape == (self.BATCH_SIZE,1, self.N)
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
        # Compute loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(self.quantile_tau -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()
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
            
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    #assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss

def eval_runs(eps, frame, env):
    """
    Makes an evaluation run with the current epsilon
    """
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

def RQ_DQN_train(env, timesteps):
    # PARAMETERS
    seed = np.random.randint(0,100000)

    alpha = 0.99 # Discount factor
    tau  = 0.005 # Target network update rate    
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

    agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        Network="DDQN",
                        layer_size=64,
                        n_step=n_step,
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
    ######