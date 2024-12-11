import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle

class HighwayEnvFastCustomReward(HighwayEnvFast):

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )

        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["safe_distance_reward"],
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


        # safe distance reward
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        safe_distance = 5
        if front_vehicle:
            distance = max(front_vehicle.position[0] - self.vehicle.position[0], 0)

            if distance > safe_distance:
                safe_distance_reward = 1  # Full reward if distance is safe
            else:
                safe_distance_reward = -1 * (safe_distance - distance) / safe_distance
        else:
            safe_distance_reward = 0


        # Improved collision reward
        collision_penalty = -1 if self.vehicle.crashed else 0  # Full penalty for collision
        near_miss_penalty = 0
        for vehicle in self.road.vehicles:
            if vehicle is not self.vehicle:
                distance_to_vehicle = np.linalg.norm(
                    np.array(self.vehicle.position) - np.array(vehicle.position)
                )
                if 0 < distance_to_vehicle <= 2:  # Near-miss threshold
                    near_miss_penalty += -0.5 * (2 - distance_to_vehicle) / 2

        collision_reward = collision_penalty + near_miss_penalty


        # high speed reward
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


        return {
            "collision_reward": collision_reward,
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": high_speed_reward,
            "on_road_reward": float(self.vehicle.on_road),
            "safe_distance_reward": safe_distance_reward,
        }
