import gym
import numpy as np
import random

from envs.frozenlake import generate_two_random_maps, generate_two_opposing_maps, generate_two_identical_maps
from envs.gridworld import GridWorld

def generate_experiment_config(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.env == 'GridWorld':
        if args.exp_name == "default":
            pass
        else:
            Exception(f'Invalid experiment name: {args.exp_name}')
        return {
            'env_configs': [
                {'flip': False}, 
                {'flip': True}
            ],
            'env_cls': GridWorld
        }
    elif args.env == 'FrozenLake':
        if args.exp_name == "randomly_heterogenous": # generate two random maps
            map1, map2 = generate_two_random_maps(size=4)
        elif args.exp_name == "strongly_heterogeous":
            map1, map2 = generate_two_opposing_maps(size=4)
        elif args.exp_name == "homogeneous":
            map1, map2 = generate_two_identical_maps(size=4)
        else:
            Exception(f'Invalid experiment name: {args.exp_name}')
        return {
            'env_configs': [
                {
                    'id': 'FrozenLake-v1',
                    'desc': map1,
                    'map_name': '4x4',
                    'is_slippery': False,
                }, 
                {
                    'id': 'FrozenLake-v1',
                    'desc': map2,
                    'map_name': '4x4',
                    'is_slippery': False,
                },
            ],
        }
    else:
        raise Exception(f'Invalid environment: {args.env}')

def create_envs(env_dict):
    env_configs = env_dict['env_configs']
    if 'env_cls' in env_dict:
        env_cls = env_dict['env_cls']
    else:
        env_cls = gym.make
    
    env_list = []
    for i, config in enumerate(env_configs):
        env = env_cls(**config)
        env.id = f'M{i+1}'
        env_list.append(env)

    return env_list