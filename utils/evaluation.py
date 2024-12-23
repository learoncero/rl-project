import pprint
import numpy as np
import gymnasium
from stable_baselines3 import DQN, PPO, A2C, SAC
import highway_env
from collections import defaultdict
from sb3_contrib import TRPO
import csv

def evaluate_model(env, model_path, algorithm='DQN', total_episodes=200, config_updates=None):
    """
    Evaluate a trained model on a given environment.

    Parameters:
    - env: Gymnasium environment.
    - config_updates: Dictionary of environment configuration updates.
    - model_path (str): Path to the trained model.
    - algorithm (str): RL algorithm used for training (e.g., 'DQN', 'PPO', 'A2C', 'SAC').
    - tensorboard_log_dir (str): Directory to save TensorBoard logs.
    - total_episodes (int): Number of episodes to evaluate.
    - log_interval (int): Interval for logging mean metrics.

    Returns:
    - dict: Aggregated evaluation metrics.
    """
    # Apply configuration updates if provided
    if config_updates:
        env.unwrapped.config.update(config_updates)

    pprint.pprint(env.unwrapped.config)

    # Define available algorithms
    algorithms = {
        'DQN': DQN,
        'PPO': PPO,
        'A2C': A2C,
        'TRPO': TRPO,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Available algorithms are: {list(algorithms.keys())}")

    # Load the model based on the specified algorithm
    AlgorithmClass = algorithms[algorithm]
    model = AlgorithmClass.load(model_path)

    for episode in range(total_episodes):
        done = truncated = False  # Reset the done and truncated flags to False
        obs, info = env.reset()  # Reset the environment and get the initial observation
        
        while not (done or truncated):  # Continue the simulation until done or truncated
            action, _states = model.predict(obs,
                                            deterministic=True)  # Get the action from the model, without exploration (deterministic)
            obs, reward, done, truncated, info = env.step(action)  # Perform the action in the environment

            env.render()  # Render the environment

def aggregate_and_normalize_rewards(file_path):
    """
    Read and aggregate rewards from the CSV file, then normalize by the number of steps.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - dict: Dictionary of normalized rewards (average reward per step).
    """
    rewards_summary = defaultdict(float)
    total_steps = 0  # Count the number of rows (steps)

    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_steps += 1  # Increment step count for each row
                for key, value in row.items():
                    rewards_summary[key] += float(value)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

    # Normalize rewards by total steps
    if total_steps > 0:
        normalized_rewards = {key: value / total_steps for key, value in rewards_summary.items()}
    else:
        print("Error: No steps logged in the file.")
        return None

    return normalized_rewards
