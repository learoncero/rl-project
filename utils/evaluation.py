import pprint
import numpy as np
import gymnasium
from stable_baselines3 import DQN
import highway_env

def evaluate_model(env_name, model_path, total_episodes=200):
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
    env = gymnasium.make(env_name, render_mode='rgb_array')

    pprint.pprint(env.unwrapped.config)

    # Load the trained model
    model = DQN.load(model_path)

    # Evaluate the model over multiple episodes
    for episode in range(total_episodes):
        done = truncated = False  # Reset the done and truncated flags to False
        obs, info = env.reset()  # Reset the environment and get the initial observation

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()