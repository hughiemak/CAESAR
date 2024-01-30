# The repository for the paper **CAESAR: Enhancing Federated RL in Heterogeneous MDPs through Convergence-Aware Sampling with Screening** submitted to AAMAS 2023 ALA Workshop

To run a gridworld experiment,
```
python main.py --env GridWorld --pdir [EXPERIMENT_DIRECTORY_NAME] --seed [SEED]
```
The experiments in the paper can be reproduced by setiing `[SEED] = 0, 1, ..., 29`.

To run a frozenlake experiment,
```
python main.py --env FrozenLake --pdir [EXPERIMENT_DIRECTORY_NAME] --exp_name [EXPERIMENT_NAME] --seed [SEED]
```
where `[EXPERIMENT_NAME]` can be `homogeneous`, `randomly_heterogenous`, or `strongly_heterogenous`. The experiments in the paper can be reproduced by setiing `[SEED] = 0, 1, ..., 29`.