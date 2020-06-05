# realtime-policy-distillation
Implementation of "Real-time Policy Distillation in Deep Reinforcement Learning" paper


# Installation

Python 3.8+ and [ray rllib](https://docs.ray.io/en/master/rllib.html) with [pytorch](https://pytorch.org/) backend are two main requirements. Full conda environment specification (and specific library versions) could be found in `environment.yml` file. 

# Code structure

In file `scripts/models.py` policy classes, that encapsulate both teacher and all student networks, are specified. In that file custom loss (which includes Q-loss, both versions of KL losses and imitation Q-loss) and custom evaluation function (which evaluates both teacher and all student networks on each iteration) are defined too. 

In file `scripts/trainer.py` the trainer is inhereted from the ray implementation of the APEX algorithm.

File `scripts/main.py` is an entry point to the training process.

File `plots.ipynb` contains the reproduction of all tables and figures from the report (And code, that can be used to calculate this values for any other trained model/game).

# How to run trainig

In order to run a training process, you should firstly create a config file (three config files, that was used for the project are presented in the folder `configs`). It should be in a yaml file format and it is a common `ray.rllib` config, so it accepts any field that can be accepted by the `ray.rllib` and used to tune the behaviour of the `ray.rllib.agents.dqn.ApexTrainer` trainer.
```python -m scripts.main --config path_to_config_file```

# Reproduce tables and figures from the project report

File `plots.ipynb` contains the bair minimum of code required to calculate results presented in the report. It works with tensorboard log files that is generated during the training process. We attach tensorboard event files, that can be accepted by [this link](https://drive.google.com/drive/folders/1uyWwqtgKi_sMvkWF6YDKVl_TsqW035bu?usp=sharing) (this files are too large for the git repo. To use them in `plots.ipynb` you should download them to your computer and specify the variable `EVENT_FILE_PATH` in the second cell of the ipynb file). 
