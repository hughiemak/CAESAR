import numpy as np
import random
import copy

class Agent:
	def __init__(self, agent_id, env, config):
		self.alpha = config['alpha']
		self.eps = config['eps']
		self.env = copy.deepcopy(env) # env.__class__(env.config)
		self.tenv = copy.deepcopy(env) # env.__class__(env.config)
		self.gamma = config['gamma'] 
		self.id = agent_id

		self.q_table = {}
		for state in range(self.env.observation_space.start, self.env.observation_space.n + self.env.observation_space.start):
			init_val = np.random.normal(config['mu'], config['sigma'])
			self.q_table[state] = {a: init_val for a in range(self.env.action_space.n)}
		self.setup()

	def best_action(self, s):
		q_vals = self.q_table[s]
		max_q_val = max(q_vals.values())
		max_actions = [k for k in q_vals if q_vals[k] == max_q_val]
		return random.choice(max_actions)

	def select_action(self, s, greedy=False):
		if random.random() > self.eps or greedy:
			return self.best_action(s)
		else:
			return random.choice(range(self.env.action_space.n))

	def setup(self):
		self.obs, _ = self.env.reset()
		self.old_q_table = copy.deepcopy(self.q_table)

	def step(self):
		action = self.select_action(self.obs)
		new_obs, reward, terminated, truncated, _ = self.env.step(action)
		self.update(self.obs, action, reward, new_obs, terminated, truncated)
		self.obs = new_obs
		if terminated or truncated:
			self.obs, _ = self.env.reset()

	def update(self, s, a, r, s_, terminated, truncated):
		curr_q = self.q_table[s][a]
		if terminated:
			target = r
		else:
			target = r + self.gamma * max(self.q_table[s_].values())
		self.q_table[s][a] = curr_q + self.alpha * (target - curr_q)

	def eval_episode(self):
		terminated, truncated = False, False
		obs, _ = self.tenv.reset()
		while not (terminated or truncated):
			action = self.select_action(obs, greedy=True)
			obs, reward, terminated, truncated, _ = self.tenv.step(action)
		return reward

	def eval(self):
		ep_return = self.eval_episode()
		return ep_return

	def update_old_q_table(self):
		self.old_q_table = copy.deepcopy(self.q_table)

	def get_old_new_q_values(self, state, action):
		q = self.q_table[state][action]
		q_old = self.old_q_table[state][action]
		return q_old, q
		

