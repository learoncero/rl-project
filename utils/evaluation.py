import pprint
import numpy as np
import gymnasium
from stable_baselines3 import DQN
from torch.utils.tensorboard import SummaryWriter
import highway_env

def evaluate_model(env_name, model_path, tensorboard_log_dir, total_episodes=200, log_interval=10):
    """
    Evaluate a trained model on a given environment.

    Parameters:
    - env_name (str): Name of the Gymnasium environment.
    - model_path (str): Path to the trained model.
    - tensorboard_log_dir (str): Directory to save TensorBoard logs.
    - total_episodes (int): Number of episodes to evaluate.
    - log_interval (int): Interval for logging mean metrics.

    Returns:
    - dict: Aggregated evaluation metrics.
    """
    # Initialize environment and TensorBoard writer
    writer = SummaryWriter(tensorboard_log_dir)
    env = gymnasium.make(env_name, render_mode='rgb_array')

    pprint.pprint(env.unwrapped.config)

    # Load the trained model
    model = DQN.load(model_path)

    # Initialize metrics
    rewards = []
    collisions = []
    safe_distances = []
    right_lane_usages = []

    # Evaluate the model over multiple episodes
    for episode in range(total_episodes):
        done = truncated = False
        obs, info = env.reset()
        episode_reward = 0
        episode_collisions = 0
        episode_safe_distance = 0
        episode_right_lane = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Track reward components
            reward_components = info.get("rewards", {})

            if reward_components.get('collision_reward', 0) < 0:
                episode_collisions += 1
            if reward_components.get('safe_distance_reward', 0) > 0:
                episode_safe_distance += 1
            if reward_components.get('right_lane_reward', 0):
                episode_right_lane += 1

            episode_reward += reward

        rewards.append(episode_reward)
        collisions.append(episode_collisions)
        safe_distances.append(episode_safe_distance)
        right_lane_usages.append(episode_right_lane)

        writer.add_scalar("Episode Reward", episode_reward, episode)

        # Compute mean metrics for better visualization
        if (episode + 1) % log_interval == 0:
            mean_reward = np.mean(rewards[-log_interval:])
            mean_collisions = np.mean(collisions[-log_interval:])
            mean_safe_distance = np.mean(safe_distances[-log_interval:])
            mean_right_lane_usage = np.mean(right_lane_usages[-log_interval:])

            writer.add_scalar("Mean Reward (Last {} Episodes)".format(log_interval), mean_reward, episode)
            writer.add_scalar("Mean Collisions (Last {} Episodes)".format(log_interval), mean_collisions, episode)
            writer.add_scalar("Mean Safe Distance (Last {} Episodes)".format(log_interval), mean_safe_distance, episode)
            writer.add_scalar("Mean Right Lane Usage (Last {} Episodes)".format(log_interval), mean_right_lane_usage, episode)


    # Final average metrics
    average_reward = np.mean(rewards)
    writer.add_scalar("Overall Average Reward", average_reward, total_episodes)

    # Close TensorBoard writer
    writer.close()

    # Return aggregated metrics
    return {
        "average_reward": average_reward,
        "total_collisions": np.sum(collisions),
        "total_safe_distances": np.sum(safe_distances),
        "total_right_lane_usages": np.sum(right_lane_usages),
    }
