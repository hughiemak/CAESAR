import gym

class Env:
	def __init__(self, config):

		self.config = config

		self.low, self.high = config['state_min'], config['state_max']
		self.lreward, self.rreward = config['lreward'], config['rreward'] 
		self.max_t = config['max_t']
		self.actions = config['actions']
		self.best_reward = max(self.lreward, self.rreward)

		self.optimal_q = None

		self.action_space = gym.spaces.Discrete(len(self.actions))
		self.observation_space = gym.spaces.Discrete(self.high - self.low + 1)

	def reset(self):
		self.x = (self.high + self.low) // 2
		self.t = 0
		return self.x, {}

	def step(self, a):
		if a == 0:
			self.x = self.x - 1 # max(self.low, self.x - 1)
		elif a == 1:
			self.x = self.x + 1 # min(self.high, self.x + 1)
		else:
			raise Exception(f"Invalid action {a}")

		terminated, truncated = False, False
		reward = 0.
		self.t += 1

		if self.x == self.low - 1:
			reward = self.lreward
			terminated = True
		elif self.x == self.high + 1:
			reward = self.rreward
			terminated = True

		if self.t == self.max_t:
			truncated = True

		return self.x, reward, terminated, truncated, {}

class GridWorld(Env):
	def __init__(self, flip=False):
		self.flip = flip
		lreward = -1. if self.flip==False else 1.
		rreward = -1 * lreward
		super().__init__({
			'state_min': 0,
			'state_max': 8,
			'lreward': lreward,
			'rreward': rreward,
			'max_t': 10,
			'actions': [0,1],
		})

	def compute_optimal_q(self, gamma):
		if self.flip==False:
			optimal_q = {}
			for state in range(self.low, self.high+1):
				optimal_q[state] = {a: 0. for a in self.actions}
			for state in range(self.low, self.high+1):
				optimal_q[state][1] = gamma ** (self.high - state) * self.rreward
			for state in range(self.low, self.high+1):
				if state == self.low:
					optimal_q[state][0] = self.lreward
				else:
					optimal_q[state][0] = gamma * optimal_q[state - 1][1]
			return optimal_q
		else:
			optimal_q = {}
			for state in range(self.low, self.high+1):
				optimal_q[state] = {a: 0. for a in self.actions}
			for state in range(self.high, self.low-1, -1):
				optimal_q[state][0] = gamma ** (state - self.low) * self.lreward
			for state in range(self.high, self.low-1, -1):
				if state == self.high:
					optimal_q[state][1] = self.rreward
				else:
					optimal_q[state][1] = gamma * optimal_q[state + 1][0]
			return optimal_q


class Env1(Env):
	def __init__(self):
		super().__init__({
			'state_min': 0,
			'state_max': 8,
			'lreward': -1.,
			'rreward': 1.,
			'max_t': 10,
			'actions': [0,1],
			# 'gamma': 0.9,
		})
		# self.id = 'env1'

	def compute_optimal_q(self, gamma):
		optimal_q = {}
		for state in range(self.low, self.high+1):
			optimal_q[state] = {a: 0. for a in self.actions}
		for state in range(self.low, self.high+1):
			optimal_q[state][1] = gamma ** (self.high - state) * self.rreward
		for state in range(self.low, self.high+1):
			if state == self.low:
				optimal_q[state][0] = self.lreward
			else:
				optimal_q[state][0] = gamma * optimal_q[state - 1][1]
		return optimal_q

class Env2(Env):
	def __init__(self):
		super().__init__({
			'state_min': 0,
			'state_max': 8,
			'lreward': 1.,
			'rreward': -1.,
			'max_t': 10,
			'actions': [0,1],
			# 'gamma': 0.9,
		})
		# self.id = 'env2'

	def compute_optimal_q(self, gamma):
		optimal_q = {}
		for state in range(self.low, self.high+1):
			optimal_q[state] = {a: 0. for a in self.actions}
		for state in range(self.high, self.low-1, -1):
			optimal_q[state][0] = gamma ** (state - self.low) * self.lreward
		for state in range(self.high, self.low-1, -1):
			if state == self.high:
				optimal_q[state][1] = self.rreward
			else:
				optimal_q[state][1] = gamma * optimal_q[state + 1][0]
		return optimal_q

