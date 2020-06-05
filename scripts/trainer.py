from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy

import ray
from ray import tune
from ray.tune import register_env

from scripts.models import loss_callback, custom_eval_fn


def my_get_policy(*args, **kwargs):
    print(f'GET POLICY:\n{args=}\n{kwargs=}\n')
    return MyPolicy


MyPolicy = DQNTorchPolicy.with_updates(
    name='MyPolicy',
    loss_fn=loss_callback,
)

MyTrainer=ApexTrainer.with_updates(
    name='MyDQN',
    get_policy_class=my_get_policy,
    default_policy=MyPolicy,
)
