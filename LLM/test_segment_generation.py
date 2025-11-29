import os
import time
import numpy as np
from dotenv import load_dotenv
from llm import Navigator

# Load environment variables
load_dotenv()

def test_segment_generation():
    print("="*70)
    print("TESTING DYNAMIC SEGMENT GENERATION")
    print("="*70)

    # 1. Initialize Navigator with a story
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    story = "Fly to the North to find the survivors."
    print(f"Mission: {story}")
    
    # Initialize navigator (this sets the target based on the story)
    navigator = Navigator(api_key=api_key, bounds=50.0, waypoint_spacing=5.0, story=story)
    
    print(f"Target Position: {navigator.target_pos}")
    print(f"Direction: {navigator.temp_direction}")

    # 2. Simulate Drone Movement
    # Start at [0,0,1]
    current_pos = [0.0, 0.0, 1.0]
    
    # We will request segments until we reach the target
    segment_count = 0
    max_segments = 5 

    while segment_count < max_segments:
        print(f"\n--- Requesting Segment {segment_count + 1} from {current_pos} ---")
        
        # Call the function we want to test
        new_waypoints = navigator.get_next_segment(current_pos)
        
        if not new_waypoints:
            print("No more waypoints returned. Reached target or error.")
            break
            
        print(f"Received {len(new_waypoints)} waypoints:")
        for wp in new_waypoints:
            print(f"  -> {wp}")
            
        # Simulate flying to the last waypoint of this segment
        last_wp = new_waypoints[-1]
        print(f"Simulating flight to last waypoint: {last_wp}")
        current_pos = last_wp
        
        # Check distance to target
        dist = np.linalg.norm(np.array(current_pos) - np.array(navigator.target_pos))
        print(f"Distance to target: {dist:.2f}m")
        
        if dist < 2.0:
            print("Target reached!")
            break
            
        segment_count += 1
        time.sleep(1) # Small pause to not hit API rate limits too hard

if __name__ == "__main__":
    test_segment_generation()
