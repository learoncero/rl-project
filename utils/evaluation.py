import pprint
import numpy as np
import gymnasium
from stable_baselines3 import DQN
import highway_env

def evaluate_model(env, config_updates, model_path, total_episodes=200):
    """
    Evaluate a trained model on a given environment.

    Parameters:
    - env: Gymnasium environment.
    - config_updates: Dictionary of environment configuration updates.
    - model_path (str): Path to the trained model.
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

    model = DQN.load(model_path)

    for episode in range(total_episodes):
        done = truncated = False  # Reset the done and truncated flags to False
        obs, info = env.reset()  # Reset the environment and get the initial observation
        
        while not (done or truncated):  # Continue the simulation until done or truncated
            action, _states = model.predict(obs,
                                            deterministic=True)  # Get the action from the model, without exploration (deterministic)
            obs, reward, done, truncated, info = env.step(action)  # Perform the action in the environment
            env.render()  # Render the environment