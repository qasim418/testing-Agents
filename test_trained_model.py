#!/usr/bin/env python3
"""
Visualize the trained random-point navigation policy with PyBullet GUI.

Usage:
    python test_trained_model.py [model_path] [num_episodes] [max_steps]
Defaults:
    model_path      = "dqn_random_point_nav"
    num_episodes    = 5
    max_steps       = 400
"""

import sys
import time
import os
from typing import Tuple

import numpy as np
import pybullet as p
from stable_baselines3 import DQN

from train_dqn import RandomPointNavEnv
from LLM.llm import Navigator


def _unwrap_reset(obs_result):
    """Handle both Gymnasium (obs, info) and Gym (obs) reset signatures."""
    if isinstance(obs_result, tuple):
        return obs_result[0]
    return obs_result


def _unwrap_step(step_result: Tuple):
    """
    Normalize step outputs to (obs, reward, terminated, truncated, info)
    regardless of Gym/Gymnasium version.
    """
    if len(step_result) == 5:
        return step_result
    obs, reward, done, info = step_result
    return obs, reward, done, False, info


def test_model(
    model_path: str = "dqn_random_point_nav",
    num_episodes: int = 5,
    max_steps_per_episode: int = 400,
):
    print("=" * 70)
    print(f"Loading trained model from: {model_path}")
    print("=" * 70)

    try:
        model = DQN.load(model_path)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}.zip")
        return

    print("\nCreating GUI environment...")
    env = RandomPointNavEnv(gui=True)
    
    # Get mission story
    mission_story = input("Enter mission story: ").strip()
    if not mission_story:
        mission_story = "Emergency beacon detected deep in the jungle to the north-east. Investigate immediately."
    
    # Initialize LLM Navigator
    navigator = Navigator(api_key=os.getenv("GOOGLE_API_KEY"), bounds=50, waypoint_spacing=5, story=mission_story)
    
    episode_rewards = []
    episode_lengths = []

    try:
        # --- DYNAMIC WAYPOINTS (From LLM) ---
        dynamic_waypoints = [[0.0, 0.0, 1.5]]  # Start position
        
        print(f"Starting with dynamic waypoints. Initial: {dynamic_waypoints}")

        for episode in range(1, num_episodes + 1):
            obs = _unwrap_reset(env.reset())
            
            # --- OVERRIDE: Force Start at First Waypoint ---
            current_wp_idx = 0
            start_pos = dynamic_waypoints[0]
            
            # Move drone to start
            p.resetBasePositionAndOrientation(env.scene.drone_id, start_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
            p.resetBaseVelocity(env.scene.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.scene.client)
            
            # Get initial next segment
            next_segment = navigator.get_next_segment(start_pos)
            if next_segment:
                dynamic_waypoints.extend(next_segment)
                print(f"Initial segment added: {next_segment}")
            
            # Set initial target to next waypoint
            target_wp_idx = 1
            if target_wp_idx < len(dynamic_waypoints):
                target_pos = np.array(dynamic_waypoints[target_wp_idx], dtype=np.float32)
            else:
                print("No waypoints available")
                continue
            
            # Update environment goal
            env.goal_pos = target_pos
            p.resetBasePositionAndOrientation(env.goal_id, target_pos.tolist(), [0, 0, 0, 1], physicsClientId=env.scene.client)
            
            print(f"\n--- Episode {episode}/{num_episodes} ---")
            print(f"Current Target: Waypoint {target_wp_idx} -> {target_pos}")

            # Frame counter for LIDAR overlay (update less frequently to avoid debug draw failures)
            frame_count = 0
            
            ep_reward = 0.0
            steps = 0
            terminated = truncated = False
            
            while not (terminated or truncated) and steps < max_steps_per_episode:
                # Update observation with new goal relative position
                # We need to manually refresh the observation because we moved the goal/drone
                obs = env._get_obs()
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))

                # Override termination for intermediate waypoints
                if terminated and target_wp_idx < len(dynamic_waypoints) - 1:
                    terminated = False

                # Check if we reached the current waypoint (Custom check)
                # The env might return 'terminated=True' if we are close to the CURRENT goal
                # But we want to keep going until the FINAL goal
                
                drone_pos = np.array(p.getBasePositionAndOrientation(env.scene.drone_id, physicsClientId=env.scene.client)[0])
                # Force altitude to 1.5m if deviated
                if abs(drone_pos[2] - 1.5) > 0.1:
                    corrected_pos = [drone_pos[0], drone_pos[1], 1.5]
                    p.resetBasePositionAndOrientation(env.scene.drone_id, corrected_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
                    drone_pos = np.array(corrected_pos)
                dist_to_wp = np.linalg.norm(drone_pos - target_pos)
                
                # Check if reached boundary (close to target)
                dist_to_target = np.linalg.norm(drone_pos - navigator.target_pos)
                if dist_to_target < 2.0:
                    print(f"🎯 REACHED BOUNDARY at {drone_pos}! Target was {navigator.target_pos}")
                    print("Pausing for 5 seconds...")
                    time.sleep(5)
                    terminated = True
                    continue
                
                if dist_to_wp < 0.5:
                    print(f"✓ Reached Waypoint {target_wp_idx} ({target_pos})")
                    
                    # Move to next waypoint
                    target_wp_idx += 1
                    if target_wp_idx >= len(dynamic_waypoints):
                        # Try to get next segment
                        time.sleep(1)  # Avoid API rate limits
                        next_segment = navigator.get_next_segment(drone_pos.tolist())
                        if next_segment:
                            dynamic_waypoints.extend(next_segment)
                            print(f"→ Added next segment: {next_segment}")
                            target_pos = np.array(dynamic_waypoints[target_wp_idx], dtype=np.float32)
                            env.goal_pos = target_pos
                            p.resetBasePositionAndOrientation(env.goal_id, target_pos.tolist(), [0, 0, 0, 1], physicsClientId=env.scene.client)
                            print(f"→ Next Target: Waypoint {target_wp_idx} -> {target_pos}")
                        else:
                            print("🎉 PATH COMPLETE!")
                            terminated = True
                    else:
                        target_pos = np.array(dynamic_waypoints[target_wp_idx], dtype=np.float32)
                        env.goal_pos = target_pos
                        p.resetBasePositionAndOrientation(env.goal_id, target_pos.tolist(), [0, 0, 0, 1], physicsClientId=env.scene.client)
                        print(f"→ Next Target: Waypoint {target_wp_idx} -> {target_pos}")
                        
                        # Prevent environment from terminating early
                        terminated = False

                if hasattr(env.scene, "_update_camera"):
                    env.scene._update_camera()
                    
                # Only render LIDAR overlay every 50 frames to reduce debug draw calls
                frame_count += 1
                if frame_count % 50 == 0:
                    if hasattr(env.scene, "client") and env.scene.client is not None:
                        try:
                            pos, orn = p.getBasePositionAndOrientation(env.scene.drone_id, physicsClientId=env.scene.client)
                            yaw = p.getEulerFromQuaternion(orn)[2]
                            lidar_hits = env._get_lidar(pos, yaw)
                            env.scene._render_lidar_overlay(pos, yaw, lidar_hits)
                        except Exception:
                            # Silently ignore LIDAR rendering failures
                            pass

                ep_reward += reward
                steps += 1
                time.sleep(0.01)

            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)
            # Check if path is complete (no more segments available)
            path_complete = (target_wp_idx >= len(dynamic_waypoints) and not navigator.get_next_segment(drone_pos.tolist()))
            status = "SUCCESS" if path_complete else "TIMEOUT/CRASH"
            print(f"Reward: {ep_reward:.2f} | Steps: {steps} | Status: {status}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if episode_rewards:
            print("\n" + "=" * 70)
            print("Episode statistics")
            print("=" * 70)
            print(f"Episodes run: {len(episode_rewards)}")
            print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"Best reward: {np.max(episode_rewards):.2f}")
            print(f"Worst reward: {np.min(episode_rewards):.2f}")

        print("\nClosing environment...")
        env.close()
        print("Done.")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "download/dqn_random_point_nav_20251128_143337"
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60000  # 10 minutes (60000 * 0.01s)

    test_model(model_path=model_path, num_episodes=num_eps, max_steps_per_episode=max_steps)

