import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

env = gym.make('SpaceInvaders-v0')

model = DQN(CnnPolicy, env, verbose=1, tensorboard_log='dqn-invaders')
model.learn(total_timesteps=25000)
model.save("deepq_inv")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_inv")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
