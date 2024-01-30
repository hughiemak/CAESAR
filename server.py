import numpy as np

class PTable:
    def __init__(self, train_config):
        self.train_config = train_config
        self.reset()
        
    def get_prob(self, i, j):
        return self.table[i, j]

    def set_prob(self, i, j, val):
        self.table[i, j] = val
        self.table[j, i] = val

    def incr_prob(self, i, j, delta):
        p = self.get_prob(i, j)
        self.set_prob(i, j, min(p + delta, 1.))

    def decr_prob(self, i, j, delta):
        p = self.get_prob(i, j)
        self.set_prob(i, j, max(p - delta, 0.))
    
    def get_probs(self, i):
        return self.table[i]

    def reset(self):
        n = np.sum(self.train_config['n_agents'])
        self.table = np.full((n, n), self.train_config['init_p'])
        for i in range(n):
            self.table[i][i] = 1. # agent i will always select itself w.p. 1

class PCounter:
    def __init__(self, train_config):
        n = np.sum(train_config['n_agents'])
        self.count = np.zeros((n, n), dtype=int)

    def incr(self, i, j):
        self.count[i, j] += 1
        self.count[j, i] += 1

class Server:
    def __init__(self, groups, train_config):
        self.groups = groups
        self.train_config = train_config
        self.agents, self.env_ids = self.get_agents_and_env_ids()
        self.p_table = PTable(train_config)
        self.p_counter = PCounter(train_config)
        self.performances = [0. for _ in self.agents]

    def get_agents_and_env_ids(self):
        groups = self.groups
        agents = []
        env_ids = []
        for _, group in groups.items():
            agents += group['agents']
            env_ids += [group['env_id']] * len(group['agents'])
        return agents, env_ids

    def perform_self_learning_step(self):
        groups = self.groups
        for _, group in groups.items():
            env_id = group['env_id']
            for k, agent in enumerate(group['agents']):
                agent.step()

    def perform_federation_step(self, t, training_results):
        # if t == 0: return
        groups, config = self.groups, self.train_config

        # fed_interval = config['fed_interval']
        # if not (t % fed_interval == 0): return
        
        agents, env_ids = self.agents, self.env_ids
        scheme = config['scheme']

        if scheme == "Self":
            for agent, env_id in zip(agents, env_ids):
                selected_agents = [agent]
                self.fit(agent, selected_agents, config, training_results)
        elif scheme == "All":
            for agent, env_id in zip(agents, env_ids):
                selected_agents = agents
                self.fit(agent, selected_agents, config, training_results)
        elif scheme == "Peers":
            for agent, env_id in zip(agents, env_ids):
                selected_agents = [agt for agt, eid in zip(agents, env_ids) if eid == env_id]
                self.fit(agent, selected_agents, config, training_results)
        elif scheme == "Sampling":
            for i, (agent, env_id) in enumerate(zip(agents, env_ids)):
                probs = self.p_table.get_probs(i)
                is_selected = np.random.random(len(probs)) < probs # select agents according to p table probabilities
                is_selected[i] = True # make sure agent i is selected
                selected_agents = [agt for j, agt in enumerate(agents) if is_selected[j]]
                self.fit(agent, selected_agents, config, training_results)
        elif scheme == "CAESAR":
            for i, (agent, env_id) in enumerate(zip(agents, env_ids)):
                probs = self.p_table.get_probs(i)
                is_selected = np.random.random(len(probs)) < probs # select agents according to p table probabilities
                is_selected = is_selected & (np.array(self.performances) > self.performances[i]) # only select agents with better (estimated) performance
                selected_agents = [agt for j, agt in enumerate(agents) if is_selected[j]]
                self.fit(agent, selected_agents, config, training_results)
        elif scheme == "Screen":
            for i, (agent, env_id) in enumerate(zip(agents, env_ids)):
                is_selected = (np.array(self.performances) > self.performances[i]) # only select agents with better (estimated) performance
                selected_agents = [agt for j, agt in enumerate(agents) if is_selected[j]]
                self.fit(agent, selected_agents, config, training_results)
        else:
            raise Exception(f"{scheme} is not a valid federation scheme.")
        training_results['fed_steps_executed'] += config['fed_steps']
    
    def fit(self, agent, selected_agents, config, training_results):
        if len(selected_agents) == 0: return
        for t in range(config['fed_steps']):
            for state in agent.q_table:
                for action in agent.q_table[state]:
                    q_aggregate = 0.
                    for selected_agent in selected_agents:
                        q_aggregate += selected_agent.q_table[state][action]
                    q_aggregate = q_aggregate / len(selected_agents)
                    grad = agent.q_table[state][action] - q_aggregate
                    agent.q_table[state][action] -= config['fed_lr'] * grad

    def reset_p_table(self, t):
        self.p_table.reset()

    def update_p_table(self, t, training_results):
        n = len(self.agents)
        # iterate over all possible pairing of agents
        # incr/decr prob based on criterion _are_possibly_converging
        for i in range(n):
            for j in range(i+1, n):
                if self._are_possibly_converging(i, j):
                    self.p_table.incr_prob(i, j, self.train_config['delta'])
                else:
                    self.p_table.decr_prob(i, j, self.train_config['delta'])
        
    def update_old_q_tables(self):
        # update each agent's old q table to current one
        for i in range(len(self.agents)):
            self.agents[i].update_old_q_table()

    def _are_possibly_converging(self, i, j):
        d_olds = []
        ds = []
        for state in self.agents[0].q_table:
            for action in self.agents[0].q_table[state]:
                q_i_old, q_i = self.agents[i].get_old_new_q_values(state, action)
                q_j_old, q_j = self.agents[j].get_old_new_q_values(state, action)
                d_old = abs(q_i_old - q_j_old)
                d = abs(q_i - q_j)
                d_olds.append(d_old)
                ds.append(d)
        crit = (np.mean(d_olds) - np.mean(ds) > self.train_config['xi'])
        if crit: 
            self.p_counter.incr(i, j)
        return crit

    def update_performances(self):
        for i, agent in enumerate(self.agents):
            self.performances[i] = agent.eval()

        

                