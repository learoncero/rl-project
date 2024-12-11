import pprint
import gymnasium
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def train_model(env_name, config_updates=None, save_path='./logs/', model_save_path='./models/', total_timesteps=20000, notebook_name=""):
    """
    Train a DQN model on a specified environment.

    :param env_name: Name of the environment to use.
    :param config_updates: Dictionary of environment configuration updates.
    :param save_path: Path to save training checkpoints.
    :param model_save_path: Path to save the final trained model.
    :param total_timesteps: Total number of timesteps to train for.
    """
    # Create the environment
    env = gymnasium.make(env_name)

    # Apply configuration updates if provided
    if config_updates:
        env.unwrapped.config.update(config_updates)

    pprint.pprint(env.unwrapped.config)

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=f"{save_path}checkpoints/{notebook_name}/",
        name_prefix='notebook_name'
    )

    # Create the DQN model
    model = DQN(
        'MlpPolicy',
        env,
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
        tensorboard_log=f"{save_path}tensorboard/{notebook_name}_Training"
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the model
    model.save(f"{model_save_path}{notebook_name}")
    print(f"Model saved to {model_save_path}model_{notebook_name}")