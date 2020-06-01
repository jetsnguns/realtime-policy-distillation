import math

from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, build_q_losses

import ray
from ray import tune


ray.init()


tune.run(
    ApexTrainer,
    stop={"episode_reward_mean": 20000},
    config={
        "env": "SpaceInvadersNoFrameskip-v4",
        "num_gpus": 1,
        "num_workers": 8,
        "num_envs_per_worker": 8,
        # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        #"use_pytorch": True,
        "framework": "tf",
        "optimizer": {
            "num_replay_buffer_shards": 1,
        },
        "hiddens": [512],
        #"model": {
        #    "custom_model": "custom_model",
        #    "custom_options": {"tau": 1.},
        #},
        "monitor": False, # For now, to speed up testing
        "dueling": False,
        "double_q": False,
        "num_atoms": 1,
        "noisy": False,
        "hiddens": False,
        "buffer_size": 1000000,
        "target_network_update_freq": 50000,
        "timesteps_per_iteration": 25000,
        "rollout_fragment_length": 20,
        "train_batch_size": 512,
        "lr": 0.0001,
        "adam_epsilon": 0.00015,
        #
        "exploration_config": {
            "final_epsilon": 0.01,
            "epsilon_timesteps": 200000,
        },
        "prioritized_replay_alpha": 0.5,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_annealing_timesteps": 2000000,
        #
        # "evaluation_interval": 1,
        # "custom_eval_function": custom_eval_fn,
        # "evaluation_num_episodes": 30,
        # "evaluation_config": {
        #     "explore": False,
        # },
        # "log_level": "DEBUG",
        "worker_side_prioritization": False,
    },
)
