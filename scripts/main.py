import math

from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, build_q_losses

import ray
from ray import tune

from scripts.models import loss_callback, custom_eval_fn


def my_get_policy(*args, **kwargs):
    print(f'GET POLICY:\n{args=}\n{kwargs=}\n')
    return MyPolicy


ray.init()


MyPolicy = DQNTorchPolicy.with_updates(
    name='MyPolicy',
    loss_fn=loss_callback,
)


# MyTrainer = DQNTrainer.with_updates(
MyTrainer=ApexTrainer.with_updates(
    name='MyDQN',
    get_policy_class=my_get_policy,
    default_policy=MyPolicy,
)

tune.run(
    MyTrainer,
    stop={"episode_reward_mean": 20000},
    config={
        "env": "SpaceInvaders-v0",
        "num_gpus": 1,
        "num_workers": 2,
        "buffer_size": 5000,
        "use_pytorch": True,
        "model": {
            "custom_model": "custom_model",
            "custom_options": {"tau": 1.},
        },
        "monitor": True,
        "dueling": False,
        "hiddens": False,
        #
        "evaluation_interval": 30,
        "custom_eval_function": custom_eval_fn,
        "evaluation_num_episodes": 100,
        #
        "worker_side_prioritization": False,
        "target_network_update_freq": 50000,
    },
    checkpoint_freq=1,
)
