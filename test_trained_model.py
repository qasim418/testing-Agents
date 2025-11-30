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


def _detect_human_pixels(image: np.ndarray):
    """Improved color/shape detection for hi-vis human props, reduces false positives."""
    if image is None or image.size == 0:
        return False, None, 0
    rgb = image[:, :, :3]
    # Detect unique magenta color (R~255, G~0, B~255)
    mask = (
        (rgb[:, :, 0] > 220)
        & (rgb[:, :, 1] < 50)
        & (rgb[:, :, 2] > 220)
    )
    count = int(np.count_nonzero(mask))
    if count < 800:
        return False, None, count
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return False, None, count
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    aspect_ratio = y_span / (x_span + 1e-5)
    if aspect_ratio < 1.2:
        return False, None, count
    center = (int(xs.mean()), int(ys.mean()))
    return True, center, count


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
    
    def add_path_marker(position, label="✓", color=[0, 1, 0]):
        """Add a visual marker at the given position to show drone progress."""
        # Add a small green sphere
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color + [0.8], physicsClientId=env.scene.client)
        marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, 
                                      basePosition=position, physicsClientId=env.scene.client)
        # Add text label slightly above
        p.addUserDebugText(label, [position[0], position[1], position[2] + 0.5], 
                           textColorRGB=color, textSize=0.5, physicsClientId=env.scene.client)
        return marker_id
    
    # Get mission story
    mission_story = input("Enter mission story: ").strip()
    if not mission_story:
        mission_story = "Emergency beacon detected deep in the jungle to the north-east. Investigate immediately."
    
    # Initialize LLM Navigator
    navigator = Navigator(api_key=os.getenv("GOOGLE_API_KEY"), bounds=50, waypoint_spacing=5, story=mission_story)
    
    episode_rewards = []
    episode_lengths = []

    try:
        # --- NEW: EXPLORATION-BASED WAYPOINT SYSTEM ---
        # main_waypoints holds the LLM waypoints (straight line)
        # Each main waypoint generates: [left, main, right, main] exploration sequence
        main_waypoints = [[0.0, 0.0, 1.5]]  # Start position
        current_main_wp_idx = 0
        current_exploration_idx = 0
        exploration_queue = []  # Will hold [left, main, right, main]
        exploration_names = ["LEFT", "CENTER", "RIGHT", "CENTER"]
        exploration_colors = [[1,0,0], [1,1,0], [1,0.5,0], [0,1,0]]  # red, yellow, orange, green
        
        def get_next_exploration_target():
            """Get the next target from exploration sequence, or None if done"""
            nonlocal current_main_wp_idx, current_exploration_idx, exploration_queue
            
            # If no active exploration sequence, generate one
            if not exploration_queue:
                if current_main_wp_idx >= len(main_waypoints):
                    return None  # All waypoints explored
                
                main_wp = main_waypoints[current_main_wp_idx]
                exploration_queue = navigator.generate_exploration_sequence(main_wp)
                current_exploration_idx = 0
                print(f"\n{'='*60}")
                print(f"🔍 EXPLORING MAIN WP {current_main_wp_idx + 1}/{len(main_waypoints)}")
                print(f"Main WP: {main_wp}")
                print(f"Exploration: [LEFT → CENTER → RIGHT → CENTER]")
                print(f"{'='*60}")
            
            # Return current exploration point
            target = exploration_queue[current_exploration_idx]
            print(f"\n📍 Target: {exploration_names[current_exploration_idx]} → {target}")
            return target
        
        def advance_exploration():
            """Move to next exploration point. Returns True if more exist."""
            nonlocal current_exploration_idx, exploration_queue, current_main_wp_idx
            
            current_exploration_idx += 1
            
            # If finished this exploration sequence
            if current_exploration_idx >= len(exploration_queue):
                print(f"\n✅ FINISHED Main WP {current_main_wp_idx + 1}")
                current_main_wp_idx += 1
                current_exploration_idx = 0
                exploration_queue = []
                
                # Try to get next main waypoint segment from LLM
                if current_main_wp_idx >= len(main_waypoints):
                    # Ask LLM for next segment
                    time.sleep(1)  # Rate limit
                    next_segment = navigator.get_next_segment(drone_pos.tolist() if 'drone_pos' in locals() else [0,0,1.5])
                    if next_segment:
                        main_waypoints.extend(next_segment)
                        print(f"→ Added LLM segment: {next_segment}")
                        return True
                    else:
                        return False  # All done
            
            return True

        # --- GET INITIAL LLM PATH BEFORE EPISODES ---
        print("\n[Mission] Requesting initial path from LLM...")
        initial_segment = navigator.get_next_segment([0.0, 0.0, 1.5])
        
        if not initial_segment:
            print("ERROR: LLM failed to generate initial path. Aborting.")
            return
        
        main_waypoints.extend(initial_segment)
        print(f"[Mission] Initial path loaded: {len(initial_segment)} waypoints")
        print(f"[Mission] Main waypoints: {main_waypoints}")
        
        # Reposition humans towards the mission target direction
        if hasattr(env.scene, 'reposition_humans') and initial_segment:
            target_direction = np.array(navigator.target_pos) - np.array([0.0, 0.0, 1.5])
            target_direction[2] = 0  # Ignore Z for direction
            target_direction = target_direction / np.linalg.norm(target_direction)  # Normalize
            env.scene.reposition_humans(target_direction.tolist())
            print(f"[Mission] Humans repositioned towards direction: {target_direction}")

        for episode in range(1, num_episodes + 1):
            obs = _unwrap_reset(env.reset())
            
            # Move drone to start
            start_pos = [0.0, 0.0, 1.5]
            p.resetBasePositionAndOrientation(env.scene.drone_id, start_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
            p.resetBaseVelocity(env.scene.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.scene.client)
            add_path_marker(start_pos, "START", [0, 0, 1])  # Blue marker for start
            
            # Reset exploration state for new episode
            current_main_wp_idx = 0
            current_exploration_idx = 0
            exploration_queue = []
            
            # Get initial target
            target_pos = get_next_exploration_target()
            if target_pos is None:
                print("No waypoints available")
                continue
            
            env.goal_pos = np.array(target_pos, dtype=np.float32)
            p.resetBasePositionAndOrientation(env.goal_id, target_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
            
            print(f"\n--- Episode {episode}/{num_episodes} ---")
            print(f"Initial Target: {target_pos}")
            
            # Frame counter for LIDAR overlay (update less frequently to avoid debug draw failures)
            frame_count = 0
            
            ep_reward = 0.0
            steps = 0
            terminated = truncated = False
            
            # Stuck detection
            stuck_counter = 0
            prev_dist_to_wp = float('inf')
            STUCK_THRESHOLD = 1000  # ~10 seconds of no progress
            
            # Path visualization
            path_points = [start_pos.copy()]
            last_path_draw_step = 0
            
            # Human detection tracking
            human_spotted = False
            mission_requires_human = "human" in mission_story.lower() or "person" in mission_story.lower()
            
            # Pause functionality
            paused = False
            print("Controls: SPACEBAR to pause/resume simulation")
            
            # --- Main Loop (Replace the entire while loop) ---
            while not (terminated or truncated) and steps < max_steps_per_episode:
                # Check for pause/resume (Spacebar)
                keys = p.getKeyboardEvents()
                if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:  # Spacebar
                    paused = not paused
                    if paused:
                        print("⏸️ SIMULATION PAUSED - Press SPACEBAR to resume")
                    else:
                        print("▶️ SIMULATION RESUMED")
                
                if paused:
                    # Still update camera and LIDAR when paused
                    if hasattr(env.scene, "_update_camera"):
                        env.scene._update_camera()
                    
                    # Render LIDAR overlay when paused
                    frame_count += 1
                    if frame_count % 50 == 0:
                        if hasattr(env.scene, "client") and env.scene.client is not None:
                            try:
                                pos, orn = p.getBasePositionAndOrientation(env.scene.drone_id, physicsClientId=env.scene.client)
                                yaw = p.getEulerFromQuaternion(orn)[2]
                                lidar_hits = env._get_lidar(pos, yaw)
                                env.scene._render_lidar_overlay(pos, yaw, lidar_hits)
                            except Exception:
                                pass
                    
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    continue
                
                obs = env._get_obs()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
                
                # Override early termination
                if terminated and current_main_wp_idx < len(main_waypoints) - 1:
                    terminated = False
                
                # Get drone position and force altitude
                drone_pos = np.array(p.getBasePositionAndOrientation(env.scene.drone_id, physicsClientId=env.scene.client)[0])
                if abs(drone_pos[2] - 1.5) > 0.1:
                    corrected_pos = [drone_pos[0], drone_pos[1], 1.5]
                    p.resetBasePositionAndOrientation(env.scene.drone_id, corrected_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
                    drone_pos = np.array(corrected_pos)
                
                # Update path visualization (draw line every 10 steps)
                path_points.append(drone_pos.tolist())
                if steps - last_path_draw_step >= 10 and len(path_points) >= 2:
                    p.addUserDebugLine(path_points[-2], path_points[-1], [0, 1, 1], 2.0, lifeTime=0)  # Cyan line for drone path
                    last_path_draw_step = steps
                
                # Human detection
                camera_frame = None
                if hasattr(env.scene, "_get_drone_camera_image"):
                    camera_frame = env.scene._get_drone_camera_image()
                detected, center, pix_count = _detect_human_pixels(camera_frame)
                if detected and not human_spotted:
                    print(f"  ✓ Human detected through camera at pixels {center} (mask={pix_count})")
                    # Save snapshot
                    try:
                        from PIL import Image
                        snap_dir = "detected_human_snaps"
                        os.makedirs(snap_dir, exist_ok=True)
                        snap_path = os.path.join(snap_dir, f"human_detected_ep{episode}_step{steps}.png")
                        Image.fromarray(camera_frame).save(snap_path)
                        print(f"  ✓ Saved snapshot: {snap_path}")
                    except Exception as e:
                        print(f"  ✗ Could not save snapshot: {e}")
                    # Print drone coordinates and detection message
                    print(f"I got human at this xyz location: {drone_pos.tolist()}")
                    human_spotted = True
                    # Terminate search immediately after detection
                    terminated = True
                    break
                
                # --- STUCK DETECTION ---
                dist_to_wp = np.linalg.norm(drone_pos - target_pos)
                if dist_to_wp < prev_dist_to_wp - 0.1:  # Making progress (>10cm)
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                prev_dist_to_wp = dist_to_wp

                # If stuck, consult LLM
                if stuck_counter > STUCK_THRESHOLD:
                    print(f"⚠️ STUCK for {stuck_counter} steps. Consulting LLM...")
                    
                    # Get LIDAR data
                    pos, orn = p.getBasePositionAndOrientation(env.scene.drone_id, physicsClientId=env.scene.client)
                    yaw = p.getEulerFromQuaternion(orn)[2]
                    lidar_hits = env._get_lidar(pos, yaw)
                    
                    # Ask LLM
                    decision = navigator.get_stuck_resolution(
                        drone_pos.tolist(), 
                        target_pos, 
                        lidar_hits, 
                        stuck_counter
                    )
                    
                    # Execute LLM advice
                    if decision.get("decision") == "skip":
                        print("🤖 LLM: SKIP waypoint")
                        if not advance_exploration():
                            terminated = True
                        else:
                            target_pos = get_next_exploration_target()
                            if target_pos is None:
                                terminated = True
                            else:
                                env.goal_pos = np.array(target_pos, dtype=np.float32)
                                p.resetBasePositionAndOrientation(env.goal_id, target_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
                                stuck_counter = 0  # Reset
                                prev_dist_to_wp = float('inf')
                    elif decision.get("decision") == "abort":
                        print("🤖 LLM: ABORT mission")
                        terminated = True
                    else:
                        print("🤖 LLM: CONTINUE")
                        stuck_counter = 0  # Reset and keep trying
                
                # Check if reached boundary target
                dist_to_target = np.linalg.norm(drone_pos - navigator.target_pos)
                if dist_to_target < 2.0:
                    print(f"🎯 REACHED FINAL BOUNDARY at {drone_pos}")
                    time.sleep(5)
                    terminated = True
                    continue
                
                # Check if reached current exploration point
                dist_to_wp = np.linalg.norm(drone_pos - target_pos)
                if dist_to_wp < 0.5:
                    print(f"✓ Reached: {target_pos}")
                    color = exploration_colors[current_exploration_idx]
                    add_path_marker(target_pos, exploration_names[current_exploration_idx], color)
                    
                    # Advance to next exploration point
                    if not advance_exploration():
                        print("🎉 ALL WAYPOINTS FULLY EXPLORED!")
                        terminated = True
                    else:
                        # Get next target
                        target_pos = get_next_exploration_target()
                        if target_pos is None:
                            terminated = True
                        else:
                            env.goal_pos = np.array(target_pos, dtype=np.float32)
                            p.resetBasePositionAndOrientation(env.goal_id, target_pos, [0, 0, 0, 1], physicsClientId=env.scene.client)
                            print(f"→ Next Target: {target_pos}")
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
            path_complete = (current_main_wp_idx >= len(main_waypoints) and not navigator.get_next_segment(drone_pos.tolist()))
            mission_success = path_complete and (not mission_requires_human or human_spotted)
            status = "SUCCESS" if mission_success else ("HUMAN_NOT_FOUND" if mission_requires_human and not human_spotted else "TIMEOUT/CRASH")
            print(f"Reward: {ep_reward:.2f} | Steps: {steps} | Status: {status} | Human spotted: {human_spotted}")
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

