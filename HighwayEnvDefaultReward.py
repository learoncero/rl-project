import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from torch.utils.tensorboard import SummaryWriter
import os
import csv

class HighwayEnvDefaultReward(HighwayEnvFast):
    def __init__(self, *args, log_rewards_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_rewards_enabled = log_rewards_enabled
        self.csv_file_path = "default_reward_log.csv"
        # Create the CSV file and write the headers if it doesn't exist
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["collision_reward", "right_lane_reward", 
                                "high_speed_reward", "on_road_reward", "safe_distance_reward",
                                "left_vehicle_overtaken_reward"])

    def _reward(self, action: Action) -> float:
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
                    self.config["collision_reward"] ,
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

        safe_distance_reward = self.get_safe_distance_reward()
        left_car_overtaken_reward = self.get_left_vehicle_overtaken_reward() 

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "safe_distance_reward": safe_distance_reward,
            "left_vehicle_overtaken_reward" : left_car_overtaken_reward,
        }

    def log_rewards(self, rewards: dict):
        """Logs rewards to a CSV file."""
        rewards_row = [rewards[key] for key in ["collision_reward", 
                                                "right_lane_reward", 
                                                "high_speed_reward", 
                                                "on_road_reward", 
                                                "safe_distance_reward",
                                                "left_vehicle_overtaken_reward"]]
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(rewards_row)

    def close(self):
        """Close the environment."""
        super().close()

    def get_safe_distance_reward(self):
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        safe_distance = 5  # Define the safe distance threshold
        if front_vehicle:
            
            
            distance = max(front_vehicle.position[0] - self.vehicle.position[0], 0)

            if distance > safe_distance:
                safe_distance_reward = 1  # Full reward if distance is safe
            else:
                safe_distance_reward = -1 * (safe_distance - distance) / safe_distance
        else:
            safe_distance_reward = 0
        return safe_distance_reward
    

    
    def get_closest_left_vehicle(self, ego_vehicle):
        closest_distance= float("inf")
        left_vehicle = None
        ego_current_position= ego_vehicle.position[0]
        for vehicle in self.road.vehicles:
                # Check if the vehicle is in a lane to the left of the ego vehicle
                if vehicle.lane_index[2] < ego_vehicle.lane_index[2]:
                    
                    # If no left_vehicle has been found yet or the current left vehicle is closer to the ego vehicle
                    distance = abs(ego_current_position-vehicle.position[0])
                    if left_vehicle is None or distance < closest_distance:
                        left_vehicle = vehicle
                        closest_distance = distance
        return left_vehicle
        

    def get_left_vehicle_overtaken_reward(self):
        ego_vehicle = self.vehicle
        
        ego_current_position = ego_vehicle.position[0]
        left_vehicle = self.get_closest_left_vehicle( ego_vehicle)
        car_overtook_left_vehicle = False
        
        if (left_vehicle):
            left_current_position = left_vehicle.position[0]

        ego_previous_position_exists =hasattr(self, "ego_vehicle_previous_position") and (self.ego_vehicle_previous_position is not None)
        left_previous_position_exists =hasattr(self, "left_vehicle_previous_position") and (self.left_vehicle_previous_position is not None)
                
        if left_vehicle and ego_previous_position_exists and left_previous_position_exists:
            left_vehicle_was_ahead_of_ego = ~(self.ego_vehicle_previous_position > self.left_vehicle_previous_position)
            left_vehicle_now_behind_ego = ego_current_position > left_current_position
            car_overtook_left_vehicle = (left_vehicle_was_ahead_of_ego and left_vehicle_now_behind_ego)
            
        self.ego_vehicle_previous_position = ego_current_position
        # Reset previous position if no left car
        if left_vehicle is None:
            self.left_vehicle_previous_position = None
        else: 
            self.left_vehicle_previous_position = left_vehicle.position[0]
           
        if (car_overtook_left_vehicle):
            return 1
        return 0