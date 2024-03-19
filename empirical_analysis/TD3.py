import copy
import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import torch.nn.functional as F
import os

device = device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

#torch.set_default_device(device)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dimension):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dimension)
		self.l2 = nn.Linear(hidden_dimension, hidden_dimension)
		self.l3 = nn.Linear(hidden_dimension, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dimension):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dimension)
        self.l2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.l3 = nn.Linear(hidden_dimension, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dimension)
        self.l5 = nn.Linear(hidden_dimension, hidden_dimension)
        self.l6 = nn.Linear(hidden_dimension, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_dimension = 256
	):

		self.actor = Actor(state_dim, action_dim, max_action,hidden_dimension).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, hidden_dimension).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	
	def select_multiple_action(self, state):
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy()


	def train(self, experiences):
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, not_done = experiences

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

class ReplayBuffer_TD3(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, eval_episodes=10):
	state, _ = env.reset(seed + 100)
	state = dict_to_features(state)
	
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = env.reset(seed + 100)
		state = dict_to_features(state)
		done  = False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = env.step(action)
			state = dict_to_features(state)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def dict_to_features(d):
    #return torch.FloatTensor(d['values']).unsqueeze(0).numpy()
    #return torch.FloatTensor([*d['values'], *d['portfolio'], d['wealth']])
    #return torch.FloatTensor([*d['values'], *d['portfolio']])
	return torch.FloatTensor([*d['values'], *d['mu'],*d['sigma'],*d['theta']])#,*d['alloc']])

def eval_policy(policy, env, seed, eval_episodes=1):
	
	avg_reward = 0.
	for ep in range(eval_episodes):
		state, _ = env.reset(seed + 100*ep)
		state = dict_to_features(state)
		done  = False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = env.step(action)
			state = dict_to_features(state)
			avg_reward += reward

	avg_reward /= eval_episodes

	#print("---------------------------------------")
	#print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	#print("---------------------------------------")
	return avg_reward

def TD3_train(env, timesteps):
	# PARAMETERS
	seed = np.random.randint(0,100000)
	alpha = 0.99 # Discount factor
	tau  = 0.005 # Target network update rate

	HIDDEN_DIM = 64

	POLICY_NOISE =  0.2 # Noise added to target policy during critic update
	MAX_ACTION = 20
	POLICY_FREQ = 2 # Frequency of delayed policy updates
	MAX_TIMESTEPS = timesteps # Max time steps to run environment
	EXPL_NOISE = 0.1 # Std of Gaussian exploration noise
	START_TIMESTEPS = 5000 #25e3 # Time steps initial random policy is used
	NOISE_CLIP =  0.5 # Range to clip target policy noise
	BATCH_SIZE = 256 # Batch size for both actor and critic
	EVAL_FREQ = 5e3  # How often (time steps) we evaluate

	SAVE_MODEL = False
	file_name = 'N1'
	print(device)

	# Set seeds
	env.seed(seed)
	env.action_space.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = 4 # env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": alpha,
		"tau": tau,
		'hidden_dimension': HIDDEN_DIM
	}

	# Initialize policy
	kwargs["policy_noise"] = POLICY_NOISE * MAX_ACTION
	kwargs["noise_clip"] = NOISE_CLIP * MAX_ACTION
	kwargs["policy_freq"] = POLICY_FREQ
	policy = TD3(**kwargs)

	#policy.load(f"./models/{policy_file}")

	replay_buffer = ReplayBuffer_TD3(state_dim, action_dim)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, env, seed)]

	state,_ = env.reset()
	state = dict_to_features(state)

	done   = False
	score = 0
	episode_timesteps = 0
	episode_num = 0

	scores = []                        # list containing scores from each episode
	scores_window = deque(maxlen=100)  # last 100 scores
	output_history = []
	episode_num = 0
	eps_start = 1
	i_episode = 1

	for t in range(int(MAX_TIMESTEPS)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < START_TIMESTEPS:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, MAX_ACTION * EXPL_NOISE, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		next_state = dict_to_features(next_state)
		#done_bool = float(done) if episode_timesteps < env.L else 0
		done_bool = float(done)
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		score += reward

		# Train agent after collecting sufficient data
		if t >= START_TIMESTEPS:
			policy.train(replay_buffer, BATCH_SIZE)

		# Evaluate episode
		if (t + 1) % EVAL_FREQ == 0:
			evaluations.append(eval_policy(policy, env, seed))
			env.reset()
			np.save(f"{os.getcwd()}/results/{file_name}", evaluations)
			if SAVE_MODEL: policy.save(f"{os.getcwd()}/models/{file_name}")
		
		if done:
			#scores_window.append(score)       # save most recent score
			scores.append(score)              # save most recent score
			#writer.add_scalar("Average100", np.mean(scores_window), frame)
			#output_history.append(np.mean(scores_window))
			eval_score = eval_policy(policy, env, seed,eval_episodes=1)
			scores_window.append(eval_score)       # save most recent score
			output_history.append(np.mean(scores_window))
			print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, t, np.mean(scores_window)), end="")
			if i_episode % 100 == 0:
				print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,t, np.mean(scores_window)))
			i_episode +=1 
			state, _ = env.reset()
			state = dict_to_features(state)
			score = 0
			episode_timesteps = 0
			episode_num += 1
	return policy, output_history