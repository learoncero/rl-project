# Data & AI 6 Reinforcement Learning Project: Autonomous Driving with Deep Reinforcement Learning

_Group 17 - Dorothy Modrall Sperling, Manuel Schönberger, Lea Roncero_

## Custom Reward Function

### Summary of the default reward function in `highway-fast-v0`

The default reward function in the highway-fast-v0 environment is designed to promote safe and efficient driving behaviors. It calculates a scalar reward for each action based on four key factors:

1. **Collision Penalty:**

   - The `collision_reward` penalizes the agent if the vehicle crashes. This is a binary penalty, either 0 (no crash) or a negative value (collision occurred).

2. **Right Lane Reward:**

   - The `right_lane_reward` rewards the agent for staying in the rightmost lane. The reward increases as the vehicle moves closer to the rightmost lane, encouraging proper lane usage.

3. **High-Speed Reward:**

   - The `high_speed_reward` incentivizes driving at higher speeds. The reward is proportional to the vehicle's forward speed, normalized within a defined speed range.

4. **On-Road Reward:**
   - The `on_road_reward` encourages the vehicle to stay on the road by multiplying the total reward by 1 if the vehicle is on the road or 0 if it is off-road. This means that if the vehicle is off-road, the total reward is also zero.

### Shortcomings of the default reward function in `highway-fast-v0`

1. **Overemphasising the rightmost lane:**

   - The `right_lane_reward` encourages staying in the rightmost lane, which may not be in line with realistic driving goals. For example, overtaking may require swerving into other lanes, which is not directly encouraged.

2. **Speed reward out of context:**

   - The `high_speed_reward` rewards speed linearly, but does not take into account traffic density. Speeding in traffic jams should be penalised.

3. **Binary Collision Penalty:**

   - The `collision_reward` is binary, providing the same penalty regardless of the severity or cause of the crash. This ignores scenarios like near-misses, which could be penalized slightly to encourage caution.

4. **Lack of a Safe Distance Mechanism:**

   - The current reward function does not provide an incentive to maintain a safe distance from the vehicle in front. However, promoting a safe following distance is a fundamental part of safe driving. It helps avoid collisions and creates smoother traffic flow.

## Changes made to the reward function

### Incorporating Safe Distance into the Reward Function

#### Implementation

1. **Identify the Front Vehicle:**

   - Use the existing `road.neighbour_vehicles(vehicle)` method to locate the vehicle ahead of the agent.

   ```python
   front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)

   ```

2. **Compute the Distance:**

   - Calculate the distance between the agent's vehicle and the identified front vehicle. If no front vehicle exists, assume the distance is infinite.

   ```python
   if front_vehicle:
      distance = max(front_vehicle.position[0] - self.vehicle.position[0], 0)
   else:
      distance = float("inf")
   ```

3. **Define the Reward:**

   - Reward the agent if the distance exceeds a safe threshold (e.g., 10 meters). Penalize the agent as the distance decreases below this threshold.

   ```python
   safe_distance = 10
   if distance > safe_distance:
      safe_distance_reward = 1
   else:
      safe_distance_reward = -1 * (safe_distance - distance) / safe_distance
   ```

#### Advantages

- Encourages the agent to maintain a safe distance, reducing the likelihood of rear-end collisions.
- Discourages tailgating, promoting safer and more realistic driving behaviors.

### Incorporating
