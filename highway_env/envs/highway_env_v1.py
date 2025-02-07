from __future__ import annotations
from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnvV1(AbstractEnv):
    """

    This is a re-implementation of the highway driving environment,

    which fixes the issue of non-forward moving ego-vehicles.



    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,

    staying on the rightmost lanes and avoiding collisions.



    !! If the vehicle's accumulated heading is out of [-0.5*pi, 0.5*pi], the episode terminates.

    Also add a negative reward for large heading angle.

    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "vehicles_count": 15,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi/3, np.pi/3],
                "acceleration_range": [-5.0, 5.0],
            },
            "lanes_count": 4,
            "vehicles_count": 20,  # Reduced from 50 for better initial learning
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,
            "ego_spacing": 2,
            "vehicles_density": 0.7,  # Reduced from 1
            "collision_reward": -1,
            "right_lane_reward": 0.1,  # Reduced from 0.4
            "high_speed_reward": 0.4,  # Reduced from 0.8
            "lane_change_reward": -0.05,
            "reward_speed_range": [10, 25],  # Modified from [15, 30]
            "stopping_penalty": -0.5,  # Reduced from -2
            "reverse_reward": -0.5,  # Modified from -1
            "normalize_reward": True,
            "offroad_terminal": True,  # Changed to True
        })
        return config
    

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        # Initialize parent class with render_mode
        super().__init__(config=config)
    
        # Store the render mode
        self.render_mode = render_mode
        
            # Delay initialization
        self._delay = 0.0
        self._delayed_frequencies = 0

    # time elapse

    def elapse(self, delay, reset_steering=False):

        self._delay = delay

        # a steering reset option may be used by the controller

        if reset_steering:

            self.unwrapped.vehicle.action["steering"] = 0

        self._delayed_frequencies = int(
            self._delay * self.unwrapped.config["simulation_frequency"]
        )

        for _ in range(self._delayed_frequencies):

            self.unwrapped.road.act()

            self.unwrapped.road.step(1 / self.unwrapped.config["simulation_frequency"])

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        if self.render_mode:
            self.render()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return super().render()
        else:
            super().render()


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

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

def _rewards(self, action: Action) -> Dict[Text, float]:
    neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]
    
    forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
    
    # Smooth heading penalty
    heading_penalty = -np.abs(self.vehicle.heading) / (0.5 * np.pi)
    
    # Smoother speed transition
    speed_reward = np.clip(scaled_speed, 0, 1)
    if forward_speed < 5:  # Gradual penalty for very low speeds
        speed_reward = forward_speed / 5
    
    # Lane centering reward - corrected version
    lane_width = self.vehicle.lane.width
    lane_pos = self.vehicle.lane.local_coordinates(self.vehicle.position)[1]  # Get lateral position
    lane_centering = -np.abs(lane_pos) / (lane_width / 2)  # Normalize by half lane width
    
    # Terminal conditions
    if abs(self.vehicle.heading) > 0.5 * np.pi or (scaled_speed < -0.1 and self.time > 1):
        self.vehicle.crashed = True

    return {
        "collision_reward": float(self.vehicle.crashed),
        "right_lane_reward": lane / max(len(neighbours) - 1, 1),
        "high_speed_reward": speed_reward,
        "on_road_reward": float(self.vehicle.on_road),
        "heading_penalty": heading_penalty,
        "lane_centering": lane_centering * 0.1,
        "stopping_penalty": -0.5 if forward_speed < 2 else 0
    }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""

        return self.vehicle.crashed or (
            self.config["offroad_terminal"] and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""

        return self.time >= self.config["duration"]
