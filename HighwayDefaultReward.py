import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from torch.utils.tensorboard import SummaryWriter
import os
import csv

class HighwayDefaultReward(HighwayEnvFast):
    def __init__(self, *args, log_rewards_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_rewards_enabled = log_rewards_enabled
        self.csv_file_path = "rewards_log.csv"
        # Create the CSV file and write the headers if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "collision_reward", "right_lane_reward", 
                                 "high_speed_reward", "on_road_reward"])
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )

        # Log rewards to CSV if logging is enabled
        if self.log_rewards_enabled:
            self.log_rewards(rewards)

        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def log_rewards(self, rewards: dict):
        """Logs rewards to a CSV file."""
        episode_number = getattr(self, 'episode_number', 0)  # Assuming you have an episode counter
        rewards_row = [episode_number] + [rewards[key] for key in ["collision_reward", 
                                                                    "right_lane_reward", 
                                                                    "high_speed_reward", 
                                                                    "on_road_reward"]]
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(rewards_row)

    def close(self):
        """Close the environment."""
        super().close()

