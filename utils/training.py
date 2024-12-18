import pprint
import gymnasium
import highway_env
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def train_model(
    env,
    config_updates=None,
    save_path='./logs/',
    model_save_path='./models/',
    total_timesteps=20000,
    session_name="",
    algorithm='DQN',
    policy='MlpPolicy',
    algorithm_kwargs=None  # Allow passing custom arguments to the algorithm
):
    """
    Train a model using the specified RL algorithm on a given environment.

    :param env: Environment to use.
    :param config_updates: Dictionary of environment configuration updates.
    :param save_path: Path to save training checkpoints.
    :param model_save_path: Path to save the final trained model.
    :param total_timesteps: Total number of timesteps to train for.
    :param session_name: Name for the training session (used in paths).
    :param algorithm: RL algorithm to use (e.g., 'DQN', 'PPO', 'A2C').
    :param policy: Policy type (e.g., 'MlpPolicy', 'CnnPolicy').
    :param algorithm_kwargs: Additional parameters for the RL algorithm.
    """

    # Apply configuration updates if provided
    if config_updates:
        env.unwrapped.config.update(config_updates)

    pprint.pprint(env.unwrapped.config)

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=f"{save_path}checkpoints/{session_name}_{algorithm}/",
        name_prefix={algorithm}
    )

    # Define available algorithms
    algorithms = {
        'DQN': DQN,
        'PPO': PPO,
        'A2C': A2C,
        'SAC': SAC,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Available algorithms are: {list(algorithms.keys())}")

    # Select the algorithm class
    AlgorithmClass = algorithms[algorithm]

    # Initialize algorithm with environment and other parameters
    algorithm_kwargs = algorithm_kwargs or {}
    model = AlgorithmClass(
        policy,
        env,
        verbose=1,
        tensorboard_log=f"{save_path}tensorboard/{session_name}_{algorithm}_Training",
        **algorithm_kwargs
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the model
    model.save(f"{model_save_path}{session_name}_{algorithm}")
    print(f"Model saved to {model_save_path}{session_name}")
