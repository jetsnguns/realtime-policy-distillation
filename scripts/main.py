import argparse
from pathlib import Path

import ray
from ray.tune import tune
import yaml

from scripts.models import custom_eval_fn
from scripts.trainer import MyTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default='./config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['custom_eval_function'] = custom_eval_fn

    ray.init()

    tune.run(
        MyTrainer,
        stop={"episode_reward_mean": 20000},
        config=config,
        checkpoint_freq=1,
        local_dir="results",
    )
