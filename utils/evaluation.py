import pprint
import numpy as np
import gymnasium
from stable_baselines3 import DQN, PPO, A2C, SAC
import highway_env

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
        'SAC': SAC,
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