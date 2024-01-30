import numpy as np
import random
import json
import pickle 
import copy
import argparse
import time

from pathlib import Path
from tqdm import tqdm
from pdb import set_trace

from agent import Agent
from server import Server
from util import dumper, create_dir_path, generate_global_info

def initialize_agents(n_agents, env, agent_config, agent_count):
	agents = [Agent(agent_count+k, env, agent_config) for k in range(n_agents)]
	agent_q_tables = {agent_count+k:{} for k in range(len(agents))}
	return agents, agent_q_tables

def eval_agents(group, env_id, t, training_results):
	eval_performance = 0
	for agent in group['agents']:
		eval_performance += agent.eval()
	if env_id not in training_results['eval_performance']: training_results['eval_performance'][env_id] = {}
	training_results['eval_performance'][env_id][t] = eval_performance / len(group['agents'])

def log_p_table(server, t, training_results):
	training_results['p_tables'][t] = copy.deepcopy(server.p_table.table)

def log_performances(server, t, training_results):
	training_results['performances'][t] = copy.deepcopy(server.performances)

def log_groups(server, groups, t, config, training_results):
	log_interval = config['log_interval']
	if t % log_interval == 0:
		for _, group in groups.items():
			env_id = group['env_id']
			# save q tables
			for k, agent in enumerate(group['agents']):
				training_results['q_tables'][env_id][agent.id][t] = copy.deepcopy(agent.q_table)
			# save evaluation of the agents
			eval_agents(group, env_id, t, training_results)
		log_p_table(server, t, training_results)
	total_t = training_results['fed_steps_executed'] + t
	training_results['total_t'][t] = total_t

def store_training_results(args, path, agent_config, training_config, training_results, env):

	Path(f'{path}').mkdir(parents=True, exist_ok=True)

	# save configs
	with open(f'{path}/agent_config.json', 'w') as f:
		json.dump(agent_config, f)
	with open(f'{path}/train_config.json', 'w') as f:
		json.dump(training_config, f, default=dumper)

	# save q tables
	if not args.no_save:
		with open(f'{path}/q_tables.pkl', 'wb') as f:
			pickle.dump(training_results['q_tables'], f)

	# save optimal q
	if hasattr(env, 'compute_optimal_q'):
		optimal_q = {}
		for env in training_config['envs']:
			opt_q = env.compute_optimal_q(agent_config['gamma'])
			env_id = env.id
			optimal_q[env_id] = opt_q
		with open(f'{path}/optimal_q.pkl', 'wb') as f:
			pickle.dump(optimal_q, f)

	# save evaluation performance
	with open(f'{path}/eval_performance.pkl', 'wb') as f:
		pickle.dump(training_results['eval_performance'], f)

	# save total_t
	with open(f'{path}/total_t.pkl', 'wb') as f:
		pickle.dump(training_results['total_t'], f)

	# save p tables
	with open(f'{path}/p_tables.pkl', 'wb') as f:
		pickle.dump(training_results['p_tables'], f)
	
	# save performances
	with open(f'{path}/performances.pkl', 'wb') as f:
		pickle.dump(training_results['performances'], f)

def update_groups_old_q_tables(groups, t, config):
	if (t % config['update_qtable_freq']) == 0:
		for _, group in groups.items():
			for _, agent in enumerate(group['agents']):
				agent.update_old_q_table()

def train(args, train_config, agent_config):

	path = train_config['path']

	groups = {}
	training_results = {
		'q_tables': {},
		'eval_performance': {},
		'fed_steps_executed': 0,
		'total_t': {},
		'p_tables': {},
		'performances': {},
	}

	# initialize the groups
	agent_count = 0
	for i, env in enumerate(train_config['envs']):
		groups[i] = {}
		n_agents = train_config['n_agents'][i]
		agents, agent_q_tables = initialize_agents(n_agents, env, agent_config, agent_count)
		agent_count += len(agents)
		groups[i]['agents'] = agents
		groups[i]['env_id'] = env.id
		training_results['q_tables'][env.id] = agent_q_tables

	# initialize server
	server = Server(groups, train_config)

	# training loop
	for t in range(train_config['training_steps']+1):
		log_groups(server, groups, t, train_config, training_results)
		server.perform_self_learning_step()
		if t > 0 and t % train_config['p_interval'] == 0:
			server.update_p_table(t, training_results)
		if t > 0 and t % train_config['r_interval'] == 0:
			server.update_performances()
		if t > 0 and t % train_config['fed_interval'] == 0:
			server.perform_federation_step(t, training_results)
			server.update_old_q_tables()
		if train_config['p_reset_interval'] != 0 and t > 0 and t % train_config['p_reset_interval'] == 0:
			server.reset_p_table(t)

	# save training results
	store_training_results(args, path, agent_config, train_config, training_results, env)

def trainer(args, scheme, global_info):

	# reproducibility
	random.seed(args.seed)
	np.random.seed(args.seed)

	AGENT_CONFIG = {
		'alpha': 0.1,
		'eps': args.eps,
		'mu': args.mu, # init q table values with N(mu, sigma)
		'sigma': args.sigma, # init q table values with N(mu, sigma)
		'gamma': 0.9,
	}

	TRAINING_CONFIG = {
		'scheme': scheme,
		'envs': global_info['envs'],
		'env_configs': global_info['env_configs'],
		'n_agents': [10, 10],
		'training_steps': args.training_steps,
		'log_interval': 100,
		'fed_interval': args.fed_interval,
		'fed_lr': args.fed_lr,
		'fed_steps': args.fed_steps,
		'seed': args.seed,
		'update_qtable_freq': 100,
		'init_p': args.init_p,
		'delta': args.delta,
		'p_interval': args.p_interval,
		'xi': args.xi,
		'p_reset_interval': args.p_reset,
		'r_interval': args.r_interval,
	}

	# compute remaining items in train config
	dir_path = create_dir_path(args)
	TRAINING_CONFIG['path'] = f"{dir_path}/{scheme}"
	TRAINING_CONFIG['env_ids'] = [env.id for env in TRAINING_CONFIG['envs']]

	train(args, TRAINING_CONFIG, AGENT_CONFIG)

def save_train_info(dir_path, global_info):
	with open(f'{dir_path}/info.json', 'w') as f:
		json.dump(global_info, f, default=dumper)

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--pdir', default="GridWorld2", type=str, help='Parent directory name.')
	parser.add_argument('--env', default="GridWorld", type=str, help='the environment to experiment.')
	parser.add_argument('--seed', default=0, type=int, help='the random seed to use.')
	parser.add_argument('--mu', default=0., type=float, help='the mean for initializing q tables.')
	parser.add_argument('--sigma', default=1., type=float, help='the std for initializing q tables.')
	parser.add_argument('--eps', default=0.1, type=float, help='the exploration rate.')
	parser.add_argument('--training_steps', default=10000, type=int, help='number of training steps.')
	parser.add_argument('--fed_interval', default=100, type=int, help='the federation interval.')
	parser.add_argument('--fed_steps', default=1, type=int, help='the federation steps.')
	parser.add_argument('--fed_lr', default=0.1, type=float, help='the federation learning rate.')
	parser.add_argument('--exp_name', default="default", type=str, help='the parent directory name.')
	parser.add_argument('--init_p', default=0., type=float, help='initial value of p table entries.')
	parser.add_argument('--p_interval', default=100, type=int, help='the interval for updating the p table.')
	parser.add_argument('--delta', default=.1, type=float, help='delta for incr/decr p table entries.')
	parser.add_argument('--xi', default=0.0001, type=float, help='xi for criterion for incr/decr p table entries.')
	parser.add_argument('--p_reset', default=0, type=int, help='the interval for reseting p table. 0 for never reseting.')
	parser.add_argument('--r_interval', default=100, type=int, help='the interval for updating agent performances.')
	parser.add_argument('--no_save', action='store_true', help='whether to save agent q tables.')
	
	args = parser.parse_args()
	dir_path = create_dir_path(args)

	global_info = generate_global_info(args)
	
	save_train_info(dir_path, global_info)

	start_time = time.time()
	for scheme in global_info['schemes']: 
		trainer(args, scheme, global_info)
	print(f"Seed {args.seed} training time --- {time.time() - start_time} seconds ---")

	

	

	







