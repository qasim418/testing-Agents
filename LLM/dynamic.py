import os
import time
import numpy as np
import pybullet as p
from dotenv import load_dotenv
from stable_baselines3 import DQN
from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

# ==============================================================================
# NAVIGATOR (LLM)
# ==============================================================================
class Navigator:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
                print("[Navigator] Connected to Gemini API")
            except Exception as e:
                print(f"[Navigator] Error: {e}")

    def plan_path(self, start_pos, target_pos, obstacles=[]):
        if not self.model:
            print("[Navigator] ERROR: No API key")
            return []
            
        prompt = f"""
You are a drone navigation AI.
Current Position: {start_pos}
Target Position: {target_pos}

Generate a flight path as a JSON list of [x,y,z] coordinates.
Rules:
1. Max 5 meters between waypoints
2. Minimum height 1.0m
3. Output ONLY the JSON list, no markdown

Obstacles: {obstacles if obstacles else "None"}
"""
        
        print("\n" + "="*70)
        print("[DEBUG] PROMPT SENT TO GEMINI:")
        print("="*70)
        print(prompt)
        print("="*70)
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            print("\n" + "="*70)
            print("[DEBUG] RAW RESPONSE FROM GEMINI:")
            print("="*70)
            print(text)
            print("="*70 + "\n")
            
            if text.startswith("```"):
                text = text.strip("`").replace("json", "").strip()
            waypoints = json.loads(text)
            print(f"[Navigator] Generated {len(waypoints)} waypoints")
            print(f"[Navigator] Waypoints: {waypoints}")
            return waypoints
        except Exception as e:
            print(f"[Navigator] Failed: {e}")
            return []

# ==============================================================================
# MAIN DEMO
# ==============================================================================
def run_dynamic_navigation():
    # --- Setup ---
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY in .env file")
        return
    
    navigator = Navigator(api_key=api_key)
    
    # --- Load RL Model ---
    model_path = "m:/IntroToAIAgentz/Project/testing/gym-pybullet-drones/results/p2p_curriculum_11.25.2025_16.31.48/single_stage_medium/best_model.zip"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please update the path to your trained model.")
        return
    
    print(f"[Pilot] Loading RL model from {model_path}")
    pilot = DQN.load(model_path)
    
    # --- Create Environment ---
    target_pos = [30, 0, 1]  # Our desired goal
    
    env = PointToPointAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 1]]),
        target_position=np.array(target_pos),  # Force our goal
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=True,
        obs=ObservationType.RGB,
        use_city_world=True,
        city_size=50,
        obstacle_density=0.5,
        randomize_target=False  # Don't randomize - use our goal
    )
    
    # --- Mission Setup ---
    start_pos = [0, 0, 1]
    replan_distance = 2.0  # Re-plan every 2 meters
    
    obs, info = env.reset()
    current_pos = start_pos.copy()
    path_lines = []  # Store line IDs for visualization
    current_waypoint_index = 0
    current_waypoints = []
    
    print("\n" + "="*60)
    print("DYNAMIC NAVIGATION DEMO")
    print("="*60)
    print(f"Start: {start_pos}")
    print(f"Goal: {target_pos}")
    print(f"Re-planning every {replan_distance}m")
    print("="*60 + "\n")
    
    # --- Main Loop ---
    step = 0
    max_steps = 5000
    last_replan_pos = np.array(current_pos)
    total_distance_traveled = 0.0
    hover_steps = 0
    is_hovering = False
    
    while step < max_steps:
        # Calculate distance traveled since last replan
        distance_since_replan = np.linalg.norm(np.array(current_pos) - last_replan_pos)
        
        # Check if we need to re-plan
        if distance_since_replan >= replan_distance or step == 0:
            print("\n" + "#"*70)
            print(f"# RE-PLANNING TRIGGERED")
            print("#"*70)
            print(f"[Distance] Traveled since last plan: {distance_since_replan:.2f}m")
            print(f"[Distance] Total distance traveled: {total_distance_traveled:.2f}m")
            print(f"[Position] Current: {[f'{x:.2f}' for x in current_pos]}")
            print(f"[Position] Goal: {target_pos}")
            print(f"[Distance] Remaining to goal: {np.linalg.norm(np.array(current_pos) - np.array(target_pos)):.2f}m")
            
            # Enter hover mode
            print("\n[Pilot] ENTERING HOVER MODE...")
            is_hovering = True
            hover_steps = 48  # Hover for 1 second (48 control steps at 48Hz)
            
            # Clear old path lines
            for line_id in path_lines:
                try:
                    p.removeUserDebugItem(line_id)
                except:
                    pass
            path_lines = []
            
            # Ask LLM for new path
            print("\n[Navigator] Requesting new path from Gemini...")
            current_waypoints = navigator.plan_path(current_pos, target_pos)
            
            if not current_waypoints:
                print("[System] Navigation failed - no path generated")
                break
            
            # Draw new path
            prev = current_pos
            for i, wp in enumerate(current_waypoints):
                line_id = p.addUserDebugLine(prev, wp, [0, 1, 0], 3, lifeTime=0)
                path_lines.append(line_id)
                prev = wp
            
            # Update environment's target to first waypoint
            if len(current_waypoints) > 0:
                next_waypoint = current_waypoints[0]
                env._target_position = np.array(next_waypoint)
                print(f"\n[Pilot] Target updated to waypoint 1: {[f'{x:.2f}' for x in next_waypoint]}")
            
            current_waypoint_index = 0
            last_replan_pos = np.array(current_pos)
            print(f"[System] New path visualized with {len(current_waypoints)} waypoints")
            print("#"*70 + "\n")
        
        # Handle hovering
        if is_hovering and hover_steps > 0:
            # Send hover action (action 0 typically means hover/stay)
            action = 0
            hover_steps -= 1
            if hover_steps == 0:
                is_hovering = False
                print("[Pilot] HOVER COMPLETE - Resuming flight\n")
        else:
            # Check if reached current waypoint
            if len(current_waypoints) > current_waypoint_index:
                current_waypoint = current_waypoints[current_waypoint_index]
                distance_to_waypoint = np.linalg.norm(np.array(current_pos) - np.array(current_waypoint))
                
                if distance_to_waypoint < 1.0:  # Reached waypoint
                    print(f"[Pilot] ✓ Reached waypoint {current_waypoint_index + 1}/{len(current_waypoints)}")
                    current_waypoint_index += 1
                    
                    # Update to next waypoint
                    if current_waypoint_index < len(current_waypoints):
                        next_waypoint = current_waypoints[current_waypoint_index]
                        env._target_position = np.array(next_waypoint)
                        print(f"[Pilot] → Next target: waypoint {current_waypoint_index + 1}: {[f'{x:.2f}' for x in next_waypoint]}")
                    else:
                        # All waypoints reached, go to final goal
                        env._target_position = np.array(target_pos)
                        print(f"[Pilot] → Final approach to GOAL: {target_pos}")
            
            # Fly using RL Pilot
            action, _ = pilot.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update current position and calculate distance
        prev_pos = np.array(current_pos)
        state = env._getDroneStateVector(0)
        current_pos = state[0:3].tolist()
        
        # Track total distance
        step_distance = np.linalg.norm(np.array(current_pos) - prev_pos)
        total_distance_traveled += step_distance
        
        # Check if reached goal
        distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        if distance_to_goal < 0.5:
            print("\n" + "🎉"*35)
            print(f"SUCCESS! Reached goal at step {step}")
            print(f"Total distance traveled: {total_distance_traveled:.2f}m")
            print(f"Straight-line distance: {np.linalg.norm(np.array(start_pos) - np.array(target_pos)):.2f}m")
            print(f"Efficiency: {(np.linalg.norm(np.array(start_pos) - np.array(target_pos)) / total_distance_traveled * 100):.1f}%")
            print("🎉"*35 + "\n")
            break
        
        # Check if episode ended
        if terminated or truncated:
            print(f"\n[System] Episode ended at step {step}")
            print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            print(f"Total distance traveled: {total_distance_traveled:.2f}m")
            break
        
        step += 1
        time.sleep(1/240)  # Real-time visualization
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    env.close()

if __name__ == "__main__":
    run_dynamic_navigation()