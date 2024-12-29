import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from torch.utils.tensorboard import SummaryWriter
import os
import csv

class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, log_performance_metrics_enabled=True, log_filename="custom_reward_log.csv", **kwargs):

        self.log_performance_metrics_enabled=log_performance_metrics_enabled

        super().__init__(*args, **kwargs)
        

        self.performance_metrics_csv_file_path = log_filename

        
        if not os.path.exists(self.performance_metrics_csv_file_path):
            with open(self.performance_metrics_csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                                 "collision_count", 
                                 "right_lane_count", 
                                 "on_road_count", 
                                 "safe_distance_count", 
                                 "left_vehicle_overtaken_count", 
                                 "abrupt_accelerations_count"
                                ])

    def _reward(self, action: Action) -> float:
        # print("Step number", self.steps)
        # print("time elapsed", self.time)
        # print("_reward is being called")
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )         

        # Log rewards to CSV if logging is enabled
        if self.log_performance_metrics_enabled:
            print("Logging metrics for step", self.steps, "and seconds elapsed", self.time)
            self.log_performance_metrics(rewards)

        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] 
                    + self.config["left_vehicle_overtaken_reward"]
                    ,
                    self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["safe_distance_reward"]
                    +self.config['smooth_driving_reward'] 
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
        high_speed_reward = self.get_high_speed_reward(scaled_speed=scaled_speed)
        left_car_overtaken_reward = self.get_left_vehicle_overtaken_reward() 
        smooth_driving_reward = self.get_smooth_driving_reward()


        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": high_speed_reward,
            "on_road_reward": float(self.vehicle.on_road),
            "safe_distance_reward": safe_distance_reward,
            "left_vehicle_overtaken_reward" : left_car_overtaken_reward, 
            "smooth_driving_reward" : smooth_driving_reward
        }

    def log_rewards(self, rewards: dict):
        """Logs rewards to a CSV file."""
        rewards_row = [rewards[key] for key in ["collision_reward", 
                                                                    "right_lane_reward", 
                                                                    "high_speed_reward", 
                                                                    "on_road_reward", 
                                                                    "safe_distance_reward",
                                                                    "left_vehicle_overtaken_reward", 
                                                                    "smooth_driving_reward"
                                                                    ]]
        with open(self.rewards_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(rewards_row)


            
            
    def log_performance_metrics(self, rewards: dict):
        collision_count=0
        right_lane_count=0
        on_road_count=0
        safe_distance_count=0
        left_vehicle_overtaken_count=0
        abrupt_accelerations_count=0
        if rewards.get('collision_reward', 0) > 0:
            collision_count += 1
            
        if rewards.get("right_lane_reward", 0 ) > 0: 
            right_lane_count += 1
            
        if rewards.get("safe_distance_reward", 0 ) >0:
            safe_distance_count += 1
            
        if rewards.get("left_vehicle_overtaken_reward", 0 ) >0:
            left_vehicle_overtaken_count += 1
            
        if rewards.get("smooth_driving_reward", 0 ) == 0:
            abrupt_accelerations_count += 1
            
        if rewards.get("on_road_reward", 0 ) > 0:
            on_road_count += 1
            
        performance_metrics_row = [
                                collision_count, 
                                right_lane_count, 
                                on_road_count, 
                                safe_distance_count, 
                                left_vehicle_overtaken_count, 
                                abrupt_accelerations_count
            ]

        with open(self.performance_metrics_csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(performance_metrics_row)
            

                                 
        

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
    

    def get_high_speed_reward(self, scaled_speed):
        # High speed reward
        traffic_radius = 10

        # Count the number of vehicles within the traffic radius
        nearby_vehicles = 0
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:
                distance = np.linalg.norm(
                    np.array(other_vehicle.position) - np.array(self.vehicle.position)
                )
                if distance < traffic_radius:
                    nearby_vehicles += 1

        # Traffic density factor: more vehicles -> higher penalty
        max_density = 10
        traffic_density_factor = max(0, 1 - nearby_vehicles / max_density)

        # Adjust high-speed reward based on traffic density
        high_speed_reward = np.clip(scaled_speed * traffic_density_factor, 0, 1)

        return high_speed_reward
    
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

    def get_smooth_driving_reward(self):
        """Calculate reward for abrupt acceleration or deceleration."""
        max_allowed_acceleration = 2.0  # Threshold for acceptable acceleration (m/s^2)

        smooth_driving_reward = 1 

        if hasattr(self, "previous_speed") and (self.previous_speed is not None):

            current_forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            speed_change = abs(current_forward_speed - self.previous_speed)
            
            if speed_change > max_allowed_acceleration:
                smooth_driving_reward = 0
                

        # Update the previous speed for the next step
        self.previous_speed = self.vehicle.speed
        return smooth_driving_reward