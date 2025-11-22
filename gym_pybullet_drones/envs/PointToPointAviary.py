import math
import os
from typing import Optional, Tuple

import numpy as np
import pybullet as p
from gymnasium import spaces
from PIL import Image

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.city_world import CityGenerator
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class PointToPointAviary(BaseRLAviary):
    """Single-drone RL environment for point-to-point navigation with discrete velocity commands."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.RGB,
        target_position: Optional[np.ndarray] = None,
        episode_len_sec: float = 12.0,
        target_tolerance: float = 0.25,
        max_xy: float = 5.0,
        max_z: float = 2.0,
        min_z: float = 0.05,
        tilt_limit_rad: float = math.pi, # Relaxed to 180 degrees
        randomize_start: bool = False,
        randomize_target: bool = True,
        min_start_target_separation: float = 0.3,
        velocity_scale: float = 1.0, # Reduced from 1.5
        use_built_in_obstacles: bool = False,
        use_city_world: bool = True,
        city_size: int = 50,
        obstacle_density: float = 1.0,
        success_snapshot_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if obs not in (ObservationType.KIN, ObservationType.RGB):
            raise ValueError("PointToPointAviary supports ObservationType.KIN or ObservationType.RGB")
        if ctrl_freq <= 0 or pyb_freq % ctrl_freq != 0:
            raise ValueError("pyb_freq must be a positive multiple of ctrl_freq")
        if target_tolerance <= 0:
            raise ValueError("target_tolerance must be positive")
        self._rng = np.random.default_rng(seed)
        self._max_xy = float(max_xy)
        self._max_z = float(max_z)
        self._min_z = float(min_z)
        self._tilt_limit = float(tilt_limit_rad)
        self._target_tolerance = float(target_tolerance)
        self._randomize_start = bool(randomize_start)
        self._randomize_target = bool(randomize_target)
        self._min_separation = float(min_start_target_separation)
        self._velocity_scale = float(velocity_scale)
        self._episode_len_sec = float(episode_len_sec)
        self.EPISODE_LEN_SEC = self._episode_len_sec
        self._use_built_in_obstacles = bool(use_built_in_obstacles)
        self._use_city_world = bool(use_city_world)
        self._city_size = int(city_size)
        self._obstacle_density = float(obstacle_density)
        self._action_dim = 4
        self._discrete_actions = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Hover
            1: np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),   # Forward (+X)
            2: np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32),  # Backward (-X)
            3: np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),   # Right (+Y)
            4: np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32),  # Left (-Y)
            5: np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),   # Up (+Z)
            6: np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32),  # Down (-Z)
        }
        self._num_actions = len(self._discrete_actions)
        self._obs_dim = 15 if obs == ObservationType.KIN else None
        self._velocity_penalty_gain = 0.05
        self._control_penalty_gain = 0.02
        self._progress_gain = 20.0
        self._success_bonus = 200.0
        self._crash_penalty = 20.0  # Increased from 10.0
        # Stronger shaping defaults (RGB-style)
        self._step_penalty = 0.01 # Increased from 0.005
        self._survival_bonus = 0.0  # Removed positive signal
        self._distance_penalty_gain = 0.05
        self._hover_penalty = 0.02
        self._distance_reduction_bonus = 0.5
        self._velocity_limit = max(self._velocity_scale * 2.0, 1.0)
        self._max_episode_steps = int(self._episode_len_sec * ctrl_freq)
        self._last_action = np.zeros((1, self._action_dim), dtype=np.float32)
        self._last_action_index = 0
        self._last_distance = 0.0
        self._prev_distance = 0.0
        self._elapsed_steps = 0
        self._marker_radius = 0.08
        self._start_marker_color = (0.1, 0.8, 0.1, 0.9)
        self._target_marker_color = (0.85, 0.1, 0.1, 0.9)
        self._marker_bodies: list[int] = []
        self._marker_debug_ids: list[int] = []
        self._success_snapshot_dir = success_snapshot_dir
        self._episode_counter = 0
        self._captured_snapshot = False
        self._last_snapshot_path: Optional[str] = None
        self._last_rgb_frame: Optional[np.ndarray] = None
        self._instance_tag = format(id(self) & 0xFFFFFFFF, "x")
        self._manual_target = target_position is not None
        self._target_position = (
            np.array(target_position, dtype=np.float32)
            if target_position is not None
            else np.array([1.0, 0.0, 1.0], dtype=np.float32)
        )
        self._start_position = (
            np.array(initial_xyzs[0], dtype=np.float32)
            if initial_xyzs is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        if initial_xyzs is None:
            initial_xyzs = np.array([self._start_position], dtype=np.float32)
        if initial_rpys is None:
            initial_rpys = np.zeros((1, 3), dtype=np.float32)
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=ActionType.VEL,
        )
        self.SPEED_LIMIT = self._velocity_scale
        # Collision shaping
        self._object_collision_penalty = 2.0

    def _addObstacles(self):
        # Always call base to ensure ground/arena are spawned if needed
        # but skip the base's built-in obstacles when using the city world
        if self._use_city_world:
            # Base class typically loads plane/arena. Avoid double planes by not calling BaseAviary._addObstacles
            super()._addObstacles()
            # Spawn procedural city objects into the current physics client
            super()._addObstacles()
            # Spawn procedural city objects into the current physics client
            CityGenerator(self.CLIENT, seed=int(self._rng.integers(0, 1_000_000))).generate(
                size=self._city_size, num_obstacles=self._obstacle_density
            )
        elif self._use_built_in_obstacles:
            BaseAviary._addObstacles(self)
        else:
            super()._addObstacles()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._elapsed_steps = 0
        start, target = self._sample_start_and_target()
        self._start_position = start
        self._target_position = target
        self.INIT_XYZS = np.tile(self._start_position, (self.NUM_DRONES, 1))
        self._last_action = np.zeros((self.NUM_DRONES, self._action_dim), dtype=np.float32)
        self._episode_counter += 1
        self._captured_snapshot = False
        self._last_snapshot_path = None
        self._last_rgb_frame = None
        obs, info = super().reset(seed=seed, options=options)
        self._clear_markers()
        self._spawn_markers()
        self._last_distance = self._distance_to_target()
        self._prev_distance = self._last_distance
        info = dict(info)
        info.update({
            "target_position": self._target_position.copy(),
            "distance_to_target": float(self._last_distance),
            "drone_position": self._getDroneStateVector(0)[0:3].copy(),
            "snapshot_path": self._last_snapshot_path,
        })
        return obs, info

    def step(self, action):
        action_index = int(action)
        if action_index not in self._discrete_actions:
            raise ValueError(f"Invalid action index {action_index}")
        command = self._discrete_actions[action_index]
        tiled_action = np.tile(command, (self.NUM_DRONES, 1))
        self._last_action = tiled_action.astype(np.float32)
        self._last_action_index = action_index
        self._prev_distance = self._last_distance
        obs, reward, terminated, truncated, info = super().step(tiled_action)
        
        if self.GUI:
            self._update_camera()

        self._elapsed_steps += 1
        info = dict(info)
        info.update({
            "target_position": self._target_position.copy(),
            "distance_to_target": float(self._last_distance),
            "drone_position": self._getDroneStateVector(0)[0:3].copy(),
            "action_index": action_index,
            "elapsed_steps": self._elapsed_steps,
            "snapshot_path": self._last_snapshot_path,
        })
        return obs, reward, bool(terminated), bool(truncated), info

    def _update_camera(self):
        if not self.GUI:
            return
        # Get current camera state to preserve user's rotation/zoom
        cam_info = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
        # cam_info: width, height, viewMatrix, projectionMatrix, cameraUp, cameraForward, horizontal, vertical, yaw, pitch, dist, target
        
        # Update target to drone's position
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        p.resetDebugVisualizerCamera(cameraDistance=cam_info[10],
                                     cameraYaw=cam_info[8],
                                     cameraPitch=cam_info[9],
                                     cameraTargetPosition=drone_pos,
                                     physicsClientId=self.CLIENT
                                     )

    def _actionSpace(self):
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, self._action_dim), dtype=np.float32))
        return spaces.Discrete(self._num_actions)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN:
            pos_bound_xy = self._max_xy * 2.0
            pos_bound_z = self._max_z
            dist_bound = math.sqrt((2 * self._max_xy) ** 2 + self._max_z ** 2)
            low_single = np.array(
                [
                    -pos_bound_xy,
                    -pos_bound_xy,
                    -pos_bound_z,
                    -self._velocity_limit,
                    -self._velocity_limit,
                    -self._velocity_limit,
                    -math.pi,
                    -math.pi,
                    -math.pi,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            high_single = np.array(
                [
                    pos_bound_xy,
                    pos_bound_xy,
                    pos_bound_z,
                    self._velocity_limit,
                    self._velocity_limit,
                    self._velocity_limit,
                    math.pi,
                    math.pi,
                    math.pi,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    dist_bound,
                    1.0,
                ],
                dtype=np.float32,
            )
            low = np.tile(low_single, (self.NUM_DRONES, 1))
            high = np.tile(high_single, (self.NUM_DRONES, 1))
            return spaces.Box(low=low, high=high, dtype=np.float32)

        height = int(self.IMG_RES[1])
        width = int(self.IMG_RES[0])
        channels = 4
        return spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN:
            obs = np.zeros((self.NUM_DRONES, self._obs_dim), dtype=np.float32)
            remaining_ratio = 1.0 - (self._elapsed_steps / max(1, self._max_episode_steps))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                position = state[0:3]
                velocity = state[10:13]
                rpy = state[7:10]
                error = self._target_position - position
                distance = np.linalg.norm(error)
                obs[i, :] = np.hstack(
                    [
                        error,
                        velocity,
                        rpy,
                        self._last_action[i],
                        np.array([distance, max(0.0, remaining_ratio)], dtype=np.float32),
                    ]
                )
            return obs.astype(np.float32)

        base_images = super()._computeObs()
        if base_images.dtype != np.uint8:
            base_images_uint8 = np.clip(base_images, 0, 255).astype(np.uint8)
        else:
            base_images_uint8 = base_images

        for i in range(self.NUM_DRONES):
            if i == 0:
                rgb_uint8 = base_images_uint8[i]
                self._last_rgb_frame = rgb_uint8[:, :, :3].copy()
        return base_images_uint8

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]
        velocity = state[10:13]
        distance = np.linalg.norm(self._target_position - position)
        progress = self._prev_distance - distance
        
        # 1. Proximity Reward (Dense, always positive)
        # Range: ~0.02 (at 50m) to 1.0 (at 0m)
        # proximity_reward = 1.0 / (1.0 + distance)
        
        reward = self._progress_gain * progress
        # reward += proximity_reward
        # reward += self._survival_bonus
        
        # reward -= self._velocity_penalty_gain * np.linalg.norm(velocity)
        # reward -= self._control_penalty_gain * np.linalg.norm(self._last_action[0, :3])
        reward -= self._step_penalty
        # reward -= self._distance_penalty_gain * distance
        # if self._last_action_index == 0:
        #     reward -= self._hover_penalty
        
        if progress > 0:
            reward += self._distance_reduction_bonus * progress
            
        # Contact-based collision shaping: penalize any physical contacts
        try:
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
            if contacts and len(contacts) > 0:
                reward -= self._object_collision_penalty
        except Exception:
            pass
        if self._is_success(distance):
            self._maybe_capture_goal_snapshot()
            reward += self._success_bonus
        if self._is_crash(position, state[7:10]):
            reward -= self._crash_penalty
        self._last_distance = distance
        return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]
        distance = np.linalg.norm(self._target_position - position)
        if self._is_success(distance):
            return True
        if self._is_crash(position, state[7:10]):
            return True
        return False

    def _computeTruncated(self):
        if self._elapsed_steps >= self._max_episode_steps:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        return {
            "target_position": self._target_position.copy(),
            "distance_to_target": float(self._last_distance),
            "drone_position": state[0:3].copy(),
            "snapshot_path": self._last_snapshot_path,
            "is_success": self._is_success(self._last_distance),
            "is_crash": self._is_crash(state[0:3], state[7:10])
        }

    def _preprocessAction(self, action):
        return super()._preprocessAction(action)

    def _sample_start_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        start = self._start_position.copy()
        if self._randomize_start:
            start = self._sample_position(z_min=self._min_z + 0.05)
        target = self._target_position.copy()
        if self._randomize_target or not self._manual_target:
            for _ in range(50):
                candidate = self._sample_position(z_min=self._min_z + 0.2)
                if np.linalg.norm(candidate - start) >= self._min_separation:
                    target = candidate
                    break
            else:
                target = start + np.array([0.4, 0.0, 0.3], dtype=np.float32)
        return start.astype(np.float32), target.astype(np.float32)

    def _sample_position(self, z_min: float = 0.1) -> np.ndarray:
        xy_limit = self._max_xy * 0.98  # keep slight margin to avoid instant boundary hits
        x = self._rng.uniform(-xy_limit, xy_limit)
        y = self._rng.uniform(-xy_limit, xy_limit)
        z = self._rng.uniform(z_min, self._max_z * 0.9)
        return np.array([x, y, z], dtype=np.float32)

    def _maybe_capture_goal_snapshot(self) -> None:
        if (
            self._success_snapshot_dir is None
            or self._captured_snapshot
            or self.OBS_TYPE != ObservationType.RGB
        ):
            return

        if self.rgb is None or len(self.rgb) == 0:
            return

        image_array = self._last_rgb_frame if self._last_rgb_frame is not None else self.rgb[0]
        if image_array is None:
            return

        os.makedirs(self._success_snapshot_dir, exist_ok=True)
        filename = f"env-{self._instance_tag}_ep{self._episode_counter:05d}_step{self._elapsed_steps:05d}.png"
        filepath = os.path.join(self._success_snapshot_dir, filename)

        if image_array.ndim == 2:
            rgb_image = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        elif image_array.shape[-1] >= 3:
            rgb_image = image_array[:, :, :3]
        else:
            rgb_image = np.repeat(image_array, 3, axis=2)

        if rgb_image.dtype != np.uint8:
            # Accept either 0-1 or 0-255 float inputs and convert to uint8 for PIL
            if np.issubdtype(rgb_image.dtype, np.floating):
                normalized = rgb_image
                if normalized.max() <= 1.0:
                    normalized = normalized * 255.0
                rgb_image = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
            else:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        if rgb_image.max() <= 1:
            rgb_image = np.clip(rgb_image * 255.0, 0.0, 255.0).astype(np.uint8)

        Image.fromarray(rgb_image).save(filepath)
        self._captured_snapshot = True
        self._last_snapshot_path = filepath

    def _distance_to_target(self) -> float:
        state = self._getDroneStateVector(0)
        return float(np.linalg.norm(self._target_position - state[0:3]))

    def _is_success(self, distance: float) -> bool:
        return distance <= self._target_tolerance

    def _is_crash(self, position: np.ndarray, rpy: np.ndarray) -> bool:
        if position[2] <= 0.01 or position[2] >= self._max_z: # Relaxed crash condition
            return True
        if abs(position[0]) > self._max_xy or abs(position[1]) > self._max_xy:
            return True
        if abs(rpy[0]) > self._tilt_limit or abs(rpy[1]) > self._tilt_limit:
            return True
        
        # Check for object collisions (excluding ground)
        try:
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
            for contact in contacts:
                # contact[2] is bodyB (the object collided with)
                # Check if it's not the ground plane (self.PLANE_ID)
                if contact[2] != self.PLANE_ID:
                    return True
        except Exception:
            pass
            
        return False

    def _clear_markers(self) -> None:
        if not (self.GUI or self.RECORD):
            return
        for body in self._marker_bodies:
            try:
                p.removeBody(body, physicsClientId=self.CLIENT)
            except Exception:
                pass
        self._marker_bodies = []
        for item in self._marker_debug_ids:
            try:
                p.removeUserDebugItem(item, physicsClientId=self.CLIENT)
            except Exception:
                pass
        self._marker_debug_ids = []

    def _spawn_markers(self) -> None:
        if not (self.GUI or self.RECORD):
            return
        start = self._start_position.astype(float)
        target = self._target_position.astype(float)
        ground_start = np.array([start[0], start[1], self._min_z], dtype=float)
        ground_target = np.array([target[0], target[1], self._min_z], dtype=float)
        radius = self._marker_radius
        start_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self._start_marker_color,
            physicsClientId=self.CLIENT,
        )
        target_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self._target_marker_color,
            physicsClientId=self.CLIENT,
        )
        start_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_shape,
            basePosition=ground_start.tolist(),
            physicsClientId=self.CLIENT,
        )
        target_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_shape,
            basePosition=ground_target.tolist(),
            physicsClientId=self.CLIENT,
        )
        self._marker_bodies.extend([start_body, target_body])
        self._marker_debug_ids.append(
            p.addUserDebugText(
                "START",
                ground_start + np.array([0.0, 0.0, radius * 1.8], dtype=float),
                textColorRGB=self._start_marker_color[:3],
                textSize=1.2,
                parentObjectUniqueId=start_body,
                physicsClientId=self.CLIENT,
            )
        )
        self._marker_debug_ids.append(
            p.addUserDebugText(
                "GOAL",
                ground_target + np.array([0.0, 0.0, radius * 1.8], dtype=float),
                textColorRGB=self._target_marker_color[:3],
                textSize=1.2,
                parentObjectUniqueId=target_body,
                physicsClientId=self.CLIENT,
            )
        )
        self._marker_debug_ids.append(
            p.addUserDebugLine(
                ground_start.tolist(),
                start.tolist(),
                lineColorRGB=self._start_marker_color[:3],
                lineWidth=2.0,
                physicsClientId=self.CLIENT,
            )
        )
        self._marker_debug_ids.append(
            p.addUserDebugLine(
                ground_target.tolist(),
                target.tolist(),
                lineColorRGB=self._target_marker_color[:3],
                lineWidth=2.0,
                physicsClientId=self.CLIENT,
            )
        )
