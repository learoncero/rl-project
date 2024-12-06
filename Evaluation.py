import pprint
import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import logging

# Register the custom environment
register(
    id='CustomRewardEnv',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')

# Update the environment configuration
env.unwrapped.config.update({
    "safe_distance_reward": 1.0,  # Set a positive weight to include this reward
})

env.unwrapped.config["lanes_count"] = 4 # Change the number of lanes to make environment more complex
env.unwrapped.config["vehicles_count"] = 60 # Increase the number of vehicles to make environment more complex
env.unwrapped.config["duration"] = 1000 # Increase the duration of the simulation to see how the agent behaves over a longer period

# Print the environment configuration
pprint.pprint(env.unwrapped.config)

# Set up logging for long-term performance analysis
logging.basicConfig(filename='evaluation_log.txt', level=logging.INFO)

# Load the trained model
model = DQN.load("highway_dqn/model")

# Set up evaluation parameters
total_episodes = 20  # Number of episodes to evaluate
total_reward = 0
episode_rewards = []
collision_count = 0
safe_distance_count = 0
right_lane_count = 0

# Evaluate the model over multiple episodes
for episode in range(total_episodes):
    done = truncated = False
    obs, info = env.reset()
    episode_reward = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        # Track reward components
        reward_components = info.get("reward_components", {})

        # Track collisions and safe distance
        if reward_components.get('collision_reward', 0) > 0:
            collision_count += 1
        if reward_components.get('safe_distance_reward', 0) > 0:
            safe_distance_count += 1
        if info.get('right_lane', False):  # Assuming `right_lane` is part of info
            right_lane_count += 1

        # Update the total reward for the episode
        episode_reward += reward

    # Update total reward and log the result
    total_reward += episode_reward
    episode_rewards.append(episode_reward)
    logging.info(f"Episode {episode + 1}: Total reward = {episode_reward}")

    # Print the results for each episode
    print(f"Episode {episode + 1}: Total reward = {episode_reward}")
    print(f"Collisions encountered: {collision_count}")
    print(f"Safe distance maintained: {safe_distance_count}")
    print(f"Right lane usage: {right_lane_count}")

# Calculate and print the average reward
average_reward = total_reward / total_episodes
print(f"Average reward over {total_episodes} episodes: {average_reward}")

# Plot the rewards over episodes
plt.plot(range(total_episodes), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

# Log the average reward in the evaluation log
logging.info(f"Average reward over {total_episodes} episodes: {average_reward}")
