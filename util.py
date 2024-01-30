import json
import inspect

from pathlib import Path
from config import generate_experiment_config, create_envs

def create_dir_path(args):
	dir_path = f"result/{args.pdir}/{args.exp_name}/mu{args.mu}/sig{args.sigma}/eps{args.eps}/fedi{args.fed_interval}/feds{args.fed_steps}/init_p{args.init_p}/p_interval{args.p_interval}/delta{args.delta}/xi{args.xi}/p_reset{args.p_reset}/seed{args.seed}"
	Path(f'{dir_path}').mkdir(parents=True, exist_ok=True)
	return dir_path

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__class__.__name__

def generate_global_info(args):
    experiment_config = generate_experiment_config(args)
    envs = create_envs(experiment_config)
    global_info = {
        'env': args.env,
        'schemes': ['Self', 'All', 'Peers', 'Sampling', 'CAESAR', 'Screen'],
		'envs': envs,
		'env_configs': experiment_config['env_configs'],
		'action_space': list(range(envs[0].action_space.n)),
        'observation_space': list(range(envs[0].observation_space.n)),
	}
    return global_info