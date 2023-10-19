import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym
from IPython.display import clear_output

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
LOC_MIN = -200
LOC_MAX = 200
#N = 1 # TODO: this needs to be removed in the future

class PolicyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        """
        This is a basic Artificial Neural Network that can be used for both the Actor and the Critic.
        """
        super().__init__()
        self.sigma_out = torch.FloatTensor(np.ones(output_dim)) * 0.5

        self.fc_1 = nn.Linear(input_dim,hidden_dim) 
        self.fc_2 = nn.Linear(hidden_dim,hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim,output_dim)
        self.fc_sigma = nn.Linear(hidden_dim,output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Prediction of the Neural Network given an input x (in RL, x is a state).
        The Network uses a dropout layer (to help generalize), and the ReLU activation function.
        """
        # first two hidden layers are shared for mu and sigma
        x = F.leaky_relu(self.fc_1(x))
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc_2(x))
        x = self.dropout(x)

        mu      = torch.clamp(self.fc_mu(x),min=LOC_MIN, max=LOC_MAX) # final layer estimates mu
        log_std = self.fc_sigma(x) # final layer estimates sigma
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # We limit the variance by forcing within a range of -2,20
        sigma   = log_std.exp()
        sigma   = self.sigma_out
        return mu, sigma

class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        """
        This is a basic Artificial Neural Network that can be used for both the Actor and the Critic.
        """
        super().__init__()

        self.fc_1 = nn.Linear(input_dim,hidden_dim) 
        self.fc_2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim,output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Prediction of the Neural Network given an input x (in RL, x is a state).
        The Network uses a dropout layer (to help generalize), and the ReLU activation function.
        """
        x = F.leaky_relu(self.fc_1(x))
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc_2(x))
        x = self.dropout(x)
        x = self.fc_3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        """
        This is a joint model, with two ANNs within.
        """
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        """ 
        The output of the ActorCritic model is the concatenation of the actor and critic's outputs.
        """

        action_mu_pred, action_sigma_pred = self.actor(state)
        value_pred   = self.critic(state)
        
        return action_mu_pred, action_sigma_pred, value_pred

def calculate_returns(rewards, values, discount_factor, normalize = True):
    """
    Function to calculate rewards in time step order and normalize them.
    Normalizing stabilizes the results.
    """
    # TODO: calculate future rewards
    returns = np.zeros(len(rewards))
    rewards = rewards + [0,0,0]
    vals    = list(values.detach().numpy()) + [0,0,0]
    for i in range(len(returns)):
        # TODO: Calculate n step return :
        #val = values[i+3] if i+4 <= len(returns) else 0
        returns[i] = rewards[i] + discount_factor * rewards[i+1] + discount_factor**2 * rewards[i+2] \
            + discount_factor**3 * vals[i+3]

    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return torch.tensor(returns)

def calculate_advantages(returns, values, normalize = True):
    """
    Computes the advantage for all actions. 
    Reminder: the Advantage function for an action a is A(s,a) = Q(s,a) - V(s)
    """
    # TODO: calculate advantage
    advantages = returns - values
    
    # TODO: write code to normalize the values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages

def update_policy(advantages, log_prob_actions, returns, values):
    """
    Function to update your policy based on your actor and critic loss.
    """
    
    advantages = advantages.detach()
    returns = returns.detach()
    
    #calculate policy loss based on advantages and log_prob_actions.
    policy_loss = -(log_prob_actions * advantages).mean()
    
    # calculate value loss based on Mean Absolute Error
    value_loss = ((returns-values)**2).mean()
        
    return policy_loss, value_loss

def train(env, agent, optimizer, discount_factor):
    """
    Performs a single training step over an episode.
    """
    agent.train()
    
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        #state = torch.FloatTensor(state[:N]).unsqueeze(0)
        state = torch.FloatTensor(state).unsqueeze(0) #TODO: change state

        action_mu, action_sigma, value_pred = agent.forward(state)
        
        dist   = distributions.Normal(action_mu, action_sigma)
        action = dist.sample()
        log_prob_action = dist.log_prob(action) # not sure if this is correct for the normal distribution
        log_prob_action = log_prob_action.sum().unsqueeze(-1)
        
        state, reward, done, _ = env.step(action.numpy())
        
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward
    
    log_prob_actions = torch.cat(log_prob_actions)
    #print('log prob: {}'.format(log_prob_actions))

    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, values, discount_factor)
    #print('returns: {}'.format(returns))

    advantages = calculate_advantages(returns, values)
    #print('adv: {}'.format(advantages))

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values)

    optimizer.zero_grad()
    
    policy_loss.backward()
    value_loss.backward()
    
    optimizer.step()

    return policy_loss.item(), value_loss.item(), episode_reward

def evaluate(env, agent, vis=False):
    """
    Function to evaluate your agent's performance.
    """
    agent.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()
    if vis: env.render()
    while not done:

        #state = torch.FloatTensor(state[:N]).unsqueeze(0)
        state = torch.FloatTensor(state).unsqueeze(0) #todo change state
        with torch.no_grad():
            action,_, _ = agent.forward(state) #TODO: Not sure if only Mu is used when performing eval

        state, reward, done, _ = env.step(action.numpy())

        episode_reward += reward

    return episode_reward

def plot(frame_idx, train_rewards, policy_loss, value_loss):
    """
    Plots the running reward and losses.
    Parameters
    ----------
    frame_idx: int
        frame id
    rewards: int
        accumulated reward
    losses: int
        loss
    """
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (frame_idx, np.mean(train_rewards[-10:])))
    plt.plot(train_rewards)
    plt.subplot(132)
    plt.title('policy loss')
    plt.plot(policy_loss)
    plt.subplot(133)
    plt.title('value loss')
    plt.plot(value_loss)
    plt.show()

def update_policy_ppo(agent, states, actions, log_prob_actions, advantages, returns, optimizer,scheduler, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        # action_prob, value_pred = agent(states)
        # value_pred = value_pred.squeeze(-1)
        # dist = distributions.Categorical(action_prob)
        
        action_mu, action_sigma, value_pred = agent.forward(states)
        value_pred = value_pred.squeeze(-1)
        dist   = distributions.Normal(action_mu, action_sigma)

        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        new_log_prob_actions = new_log_prob_actions.sum(axis=1)
        # TODO calculate policy ratio
        policy_ratio =  new_log_prob_actions - log_prob_actions
        
        # TODO calculate policy_loss_1, part of actor's loss          
        policy_loss_1 = policy_ratio * advantages 
        
        # TODO Calculate clipped part of actor's loss. Hint: check torch clamp.
        policy_loss_2 = torch.clamp(policy_ratio,1-ppo_clip,1+ppo_clip) * advantages
        
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
        #policy_loss = - torch.min(policy_loss_1.sum(), policy_loss_2.sum())

        # TODO calculate value_loss
        value_loss = ((returns-value_pred)**2).mean()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    scheduler.step()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def train_ppo(env, agent, optimizer,scheduler, discount_factor, ppo_steps, ppo_clip):
        
    agent.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        #state = torch.FloatTensor(state[:N]).unsqueeze(0)  
        state = torch.FloatTensor(state).unsqueeze(0)  #TODO: change state

        #append state here, not after we get the next state from env.step()
        states.append(state)
        
        action_mu, action_sigma, value_pred = agent.forward(state)
        dist   = distributions.Normal(action_mu, action_sigma)
        action = dist.sample()
        log_prob_action = dist.log_prob(action) # not sure if this is correct for the normal distribution
        log_prob_action = log_prob_action.sum().unsqueeze(-1)

        state, reward, done, _ = env.step(action.numpy())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, values, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy_ppo(agent, states, actions, log_prob_actions, advantages, returns, optimizer,scheduler, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward