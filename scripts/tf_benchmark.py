import ray
from ray.rllib.agents.dqn import ApexTrainer
from ray.tune import tune

ray.init()


tune.run(
    ApexTrainer,
    stop={"episode_reward_mean": 20000},
    config={
        "env": "SpaceInvaders-v0",
        "num_gpus": 1,
        "num_workers": 8,
        # "monitor": True,
        # "dueling": False,
        # "hiddens": False,
        #
        "evaluation_interval": 30,
        # "custom_eval_function": custom_eval_fn,
        # "evaluation_num_episodes": 100,
        #
        # "worker_side_prioritization": False,
        # "target_network_update_freq": 50000,
    },
    checkpoint_freq=1,
)
