import pprint

import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model_dqn')

# Uncomment the following line to register the custom environment with custom reward function, you can change the reward function in HighwayEnvCustomReward.py
register(
  id='CustomRewardEnv',
  entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')

env.unwrapped.config.update({
    "safe_distance_reward": 0.1,
})

env.unwrapped.config["lanes_count"] = 4 # Change the number of lanes to make environment more complex
env.unwrapped.config["vehicles_count"] = 60 # Increase the number of vehicles to make environment more complex
env.unwrapped.config["duration"] = 1000 # Increase the duration of the simulation to see how the agent behaves over a longer period

pprint.pprint(env.unwrapped.config)

# When using different algorithms, don't care too much about the hyperparameters, you can use the default values
model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/")

model.learn(int(20000), callback=checkpoint_callback)
model.save("highway_dqn/model")
