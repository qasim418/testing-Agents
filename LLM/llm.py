import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. NAVIGATION CONSTANTS
# ==============================================================================
DIRECTION_VECTORS = {
    "north-east": [1, 1, 0],
    "north-west": [1, -1, 0],
    "south-east": [-1, 1, 0],
    "south-west": [-1, -1, 0],
    "north": [1, 0, 0],
    "south": [-1, 0, 0],
    "east": [0, 1, 0],
    "west": [0, -1, 0]
}

# ==============================================================================
# 2. NAVIGATOR (LLM-POWERED STORY PARSER)
# ==============================================================================
class Navigator:
    @staticmethod
    def suggest_escape_direction(lidar_hits, angle_range=None, min_clearance=2.0):
        """
        Given lidar_hits (list of distances), find the direction with the largest gap/clearance.
        Assumes angles are evenly spaced from 0 to 2*pi.
        Returns the angle (in radians) to move for escape, or None if no safe direction found.
        Optionally restrict to angle_range (tuple: min, max in radians).
        """
        import math
        if len(lidar_hits) == 0:
            return None
        num_rays = len(lidar_hits)
        angles = [i * 2 * math.pi / num_rays for i in range(num_rays)]
        # Filter for clear directions
        safe = [(lidar_hits[i], angles[i]) for i in range(num_rays) if lidar_hits[i] > min_clearance]
        if not safe:
            # If no direction > min_clearance, use all directions and pick the best
            safe = [(lidar_hits[i], angles[i]) for i in range(num_rays)]
        if angle_range:
            safe = [(dist, ang) for dist, ang in safe if angle_range[0] <= ang <= angle_range[1]]
        if not safe:
            return None
        # Pick the direction with the maximum clearance
        best = max(safe, key=lambda x: x[0])
        return best[1]

    """
    Story-driven drone navigator using Gemini 2.0 Flash Lite.
    Forces waypoints to REACH THE BOUNDARY TARGET.
    """
    def __init__(self, api_key=None, bounds=50.0, waypoint_spacing=5.0):
        self.bounds = bounds
        self.spacing = waypoint_spacing
        self.api_key = api_key
        self.model = None
        
    def __init__(self, api_key=None, bounds=50.0, waypoint_spacing=5.0, story=""):
        self.bounds = bounds
        self.spacing = waypoint_spacing
        self.api_key = api_key
        self.model = None
        self.story = story
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            print(f"[Navigator] Initialized with Gemini 2.0 Flash Lite")
            print(f"[Navigator] Bounds: ±{bounds}m | Target spacing: {waypoint_spacing}m")
            
            if story:
                self.temp_direction = self._extract_direction(story)
                self.target_pos = self._calculate_target([0.0, 0.0, 1.0], self.temp_direction)
                print(f"[Navigator] Story: {story}")
                print(f"[Navigator] Direction: {self.temp_direction} | Target: {self.target_pos}")
        else:
            raise ValueError("[Navigator] No API key provided. Set GOOGLE_API_KEY environment variable.")

    def plan_path(self, start_pos, story, num_waypoints=10):
        """
        Generate waypoints from natural language story that REACH THE BOUNDARY.
        
        Args:
            start_pos: Starting position [x, y, z]
            story: Natural language mission description
            num_waypoints: Number of waypoints to generate
        
        Returns:
            tuple: (waypoints_list, target_pos, extracted_direction)
        """
        print("\n" + "="*70)
        print("[MISSION STORY]")
        print("="*70)
        print(f"\"{story}\"")
        print("="*70)

        # Extract direction and calculate target
        temp_direction = self._extract_direction(story)
        target_pos = self._calculate_target(start_pos, temp_direction)
        
        # Set instance attributes for later use
        self.temp_direction = temp_direction
        self.target_pos = target_pos
        total_distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(start_pos))
        
        print(f"[Navigator] Calculated target: {target_pos}")
        print(f"[Navigator] Total distance to target: {total_distance_to_target:.2f}m")
        
        # Segment-based planning: generate waypoints in 20m segments
        waypoints = []
        current_pos = start_pos.copy()
        remaining = total_distance_to_target
        segment_length = 20.0
        segment_count = 0
        
        while remaining > 0.1 and segment_count < 10:  # Safety limit
            # Calculate max possible distance without exceeding bounds
            vec = DIRECTION_VECTORS[temp_direction]
            max_distances = []
            for i, comp in enumerate(vec[:2]):
                if comp > 0:
                    max_distances.append((self.bounds - current_pos[i]) / comp)
                elif comp < 0:
                    max_distances.append((current_pos[i] + self.bounds) / abs(comp))
                else:
                    max_distances.append(float('inf'))
            max_segment = min(max_distances) if max_distances else float('inf')
            
            segment_dist = min(segment_length, remaining, max_segment)
            if segment_dist <= 0:
                break
            
            num_wp = max(2, int(segment_dist / self.spacing) + 1)
            is_last_segment = remaining <= segment_length + 0.1
            
            prompt = f"""
You are an emergency drone navigator. Current mission segment {segment_count + 1}.

Story: "{story}"

Current position: {current_pos}
Direction: {temp_direction}
Direction vector: {DIRECTION_VECTORS[temp_direction]}

To advance {self.spacing}m in direction {temp_direction}, add approximately {self.spacing / (2**0.5):.4f} to both X and Y coordinates (for north-east).

Generate EXACTLY {num_wp} waypoints for the next {segment_dist:.1f}m segment.

First waypoint must be: [{current_pos[0] + DIRECTION_VECTORS[temp_direction][0] * self.spacing:.4f}, {current_pos[1] + DIRECTION_VECTORS[temp_direction][1] * self.spacing:.4f}, {current_pos[2]:.1f}]
Each subsequent waypoint advances another {self.spacing}m in the same direction.

Last waypoint advances {segment_dist:.1f}m total from current position.

Maintain altitude Z = {start_pos[2]}m.
Stay within ±{self.bounds}m boundaries.

Return ONLY valid JSON: {{"waypoints": [[x1,y1,z1], [x2,y2,z2], ...]}}

RULES:
- No markdown ``` markers
- No extra text
- Waypoints must progress exactly in the direction
- Use the exact coordinates provided for the first waypoint
"""

            print(f"\n[Navigator] Segment {segment_count + 1}: Planning {segment_dist:.1f}m from {current_pos}")
            
            try:
                response = self.model.generate_content(prompt)
                raw_text = response.text.strip()
                
                print(f"[DEBUG] Segment {segment_count + 1} response:")
                print(raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
                
                # Clean JSON
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`").replace("json", "").strip()
                
                data = json.loads(raw_text)
                waypoints_segment = data.get("waypoints", [])
                # Force correct altitude to prevent crashes
                waypoints_segment = [[wp[0], wp[1], start_pos[2]] for wp in waypoints_segment]
                
                # Filter out waypoints that exceed bounds
                waypoints_segment = [wp for wp in waypoints_segment if np.all(np.abs(np.array(wp)) <= self.bounds)]
                
                if waypoints_segment:
                    if not waypoints:
                        waypoints = waypoints_segment
                    else:
                        waypoints.extend(waypoints_segment)
                    
                    current_pos = waypoints_segment[-1]
                    covered_distance = (len(waypoints_segment) - 1) * self.spacing
                    remaining -= covered_distance
                    
                    print(f"[Navigator] Segment {segment_count + 1} complete. Covered: {covered_distance:.1f}m, Remaining: {remaining:.1f}m")
                    print(f"[Navigator] Current waypoints accumulated: {len(waypoints)} total")
                    
                    # Check if close enough to target
                    dist_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                    if dist_to_target < 2.0:
                        print(f"[Navigator] Reached within {dist_to_target:.2f}m of target. Stopping path generation.")
                        break
                else:
                    print(f"[Navigator] Segment {segment_count + 1} failed: No waypoints generated")
                    break
                    
            except Exception as e:
                print(f"[Navigator] Segment {segment_count + 1} failed: {e}")
                print(f"[Navigator] Retrying segment {segment_count + 1}...")
                # Retry once
                try:
                    response = self.model.generate_content(prompt)
                    raw_text = response.text.strip()
                    
                    print(f"[DEBUG] Retry Segment {segment_count + 1} response:")
                    print(raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
                    
                    if raw_text.startswith("```"):
                        raw_text = raw_text.strip("`").replace("json", "").strip()
                    
                    data = json.loads(raw_text)
                    waypoints_segment = data.get("waypoints", [])
                    
                    # Filter out waypoints that exceed bounds
                    waypoints_segment = [wp for wp in waypoints_segment if np.all(np.abs(np.array(wp)) <= self.bounds)]
                    
                    if waypoints_segment:
                        if not waypoints:
                            waypoints = waypoints_segment
                        else:
                            waypoints.extend(waypoints_segment)
                        
                        current_pos = waypoints_segment[-1]
                        covered_distance = (len(waypoints_segment) - 1) * self.spacing
                        remaining -= covered_distance
                        
                        print(f"[Navigator] Retry Segment {segment_count + 1} complete. Covered: {covered_distance:.1f}m, Remaining: {remaining:.1f}m")
                        print(f"[Navigator] Current waypoints accumulated: {len(waypoints)} total")
                    else:
                        print(f"[Navigator] Retry Segment {segment_count + 1} failed: No waypoints generated")
                        break
                        
                except Exception as e2:
                    print(f"[Navigator] Retry Segment {segment_count + 1} also failed: {e2}")
                    break
            
            segment_count += 1
        
        # Final validation
        if waypoints:
            final_dist = np.linalg.norm(np.array(waypoints[-1]) - np.array(target_pos))
            print(f"\n[Navigator] Final waypoint distance to target: {final_dist:.2f}m")
            if final_dist > 5.0:
                print(f"[Navigator WARNING] Final waypoint is {final_dist:.2f}m from target (>5m threshold)")
        
        print(f"[Navigator] Success! Direction: {temp_direction} | Total waypoints: {len(waypoints)}")
        print(f"[Navigator] Waypoints: {waypoints}")
        return waypoints, target_pos, temp_direction

    def get_next_segment(self, current_pos):
        """Generate next 20m segment from current position towards target."""
        if not hasattr(self, 'target_pos'):
            return []
        
        remaining = np.linalg.norm(np.array(self.target_pos) - np.array(current_pos))
        if remaining < 2.0:
            return []  # Close enough to target
        
        segment_length = remaining  # Generate full remaining path in one go
        segment_dist = min(segment_length, remaining)
        
        # Calculate max possible without exceeding bounds
        vec = DIRECTION_VECTORS[self.temp_direction]
        max_distances = []
        for i, comp in enumerate(vec[:2]):
            if comp > 0:
                max_distances.append((self.bounds - current_pos[i]) / comp)
            elif comp < 0:
                max_distances.append((current_pos[i] + self.bounds) / abs(comp))
            else:
                max_distances.append(float('inf'))
        max_segment = min(max_distances) if max_distances else float('inf')
        segment_dist = min(segment_dist, max_segment)
        
        if segment_dist <= 0:
            return []
        
        num_wp = max(2, int(segment_dist / self.spacing) + 1)
        
        print(f"[LLM Input] Current position: {current_pos}")
        print(f"[LLM Input] Target position: {self.target_pos}")
        print(f"[LLM Input] Direction: {self.temp_direction}")
        print(f"[LLM Input] Direction vector: {vec}")
        print(f"[LLM Input] Segment distance: {segment_dist:.1f}m")
        print(f"[LLM Input] Number of waypoints: {num_wp}")
        print(f"[LLM Input] Spacing: {self.spacing}")
        print(f"[LLM Input] Bounds: {self.bounds}")
        print(f"[LLM Input] Story: {self.story}")
        
        # Retry up to 3 times if validation fails
        for attempt in range(3):
            prompt = f"""
You are an emergency drone navigator. Full remaining path.

Story: "{self.story}"

Current position: {current_pos}
Direction: {self.temp_direction}
Direction vector: {vec}

Generate EXACTLY {num_wp} DISTINCT waypoints for the full {segment_dist:.1f}m to the boundary.

The direction vector {vec} means for each step, add {vec[0]} to x, {vec[1]} to y, {vec[2]} to z.

First waypoint must be: [{current_pos[0] + vec[0] * self.spacing:.4f}, {current_pos[1] + vec[1] * self.spacing:.4f}, 1.5]

Each subsequent waypoint advances exactly {self.spacing}m in the same direction: add {vec[0] * self.spacing} to x, {vec[1] * self.spacing} to y, {vec[2] * self.spacing} to z.

Last waypoint advances {segment_dist:.1f}m total from current position.

Maintain altitude Z = 1.5m.
Stay within ±{self.bounds}m boundaries.

Return ONLY valid JSON: {{"waypoints": [[x1,y1,z1], [x2,y2,z2], ...]}}

RULES:
- No markdown ``` markers
- No extra text
- Waypoints must progress exactly in the direction vector
- Use the exact coordinates provided for the first waypoint
- All waypoints must be unique and distinct
- Do not repeat any waypoint
"""
            
            print(f"[LLM Prompt] Attempt {attempt + 1}:")
            print(prompt)
            
            try:
                response = self.model.generate_content(prompt)
                raw_text = response.text.strip()
                
                print(f"[LLM Output] Attempt {attempt + 1}:")
                print(raw_text)
                
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`").replace("json", "").strip()
                
                data = json.loads(raw_text)
                waypoints_segment = data.get("waypoints", [])
                # Force correct altitude to prevent crashes
                waypoints_segment = [[wp[0], wp[1], 1.5] for wp in waypoints_segment]
                # Filter out waypoints that exceed bounds
                waypoints_segment = [wp for wp in waypoints_segment if np.all(np.abs(np.array(wp)) <= self.bounds)]
                # Remove duplicates
                seen = set()
                waypoints_segment = [wp for wp in waypoints_segment if tuple(wp) not in seen and (seen.add(tuple(wp)) or True)]
                
                print(f"[DEBUG] Generated segment: {data.get('waypoints', [])}")
                print(f"[DEBUG] After processing: {waypoints_segment}")
                
                if waypoints_segment:
                    # Validate segment
                    if self._validate_segment(waypoints_segment, current_pos, vec, self.spacing):
                        return waypoints_segment
                    else:
                        print(f"[Navigator] Segment validation failed on attempt {attempt + 1}, retrying...")
                else:
                    print(f"[Navigator] No waypoints generated on attempt {attempt + 1}, retrying...")
                    
            except Exception as e:
                print(f"[Navigator] Failed to get next segment on attempt {attempt + 1}: {e}")
        
        print("[Navigator] All attempts failed, returning empty segment")
        return []

    def _validate_segment(self, waypoints, start_pos, direction_vec, spacing):
        """Quick validation for generated segment."""
        if len(waypoints) < 2:
            return False
        
        # Check no duplicates
        wp_set = set(tuple(wp) for wp in waypoints)
        if len(wp_set) < len(waypoints):
            return False
        
        # Check first waypoint
        expected_first = [start_pos[0] + direction_vec[0] * spacing, 
                         start_pos[1] + direction_vec[1] * spacing, 
                         start_pos[2]]
        if np.linalg.norm(np.array(waypoints[0]) - np.array(expected_first)) > 2.0:
            return False
        
        # Check progression
        for i in range(1, len(waypoints)):
            prev = np.array(waypoints[i-1])
            curr = np.array(waypoints[i])
            diff = curr - prev
            expected_diff = np.array(direction_vec) * spacing
            if np.linalg.norm(diff[:2] - expected_diff[:2]) > 2.0 or abs(diff[2]) > 0.1:
                return False
        
        return True

    def _extract_direction(self, story):
        """Extract direction from story keywords."""
        import re
        story_lower = story.lower()
        for direction in DIRECTION_VECTORS.keys():
            if re.search(r'\b' + re.escape(direction) + r'\b', story_lower):
                return direction
        return "north"  # Default fallback

    def _calculate_target(self, start, direction):
        """Calculate boundary target CLOSE TO EDGE."""
        vec = DIRECTION_VECTORS.get(direction.lower(), [1, 0, 0])
        
        # Find limiting distance per axis
        distances = []
        for i, comp in enumerate(vec[:2]):
            if comp > 0:
                distances.append(self.bounds - start[i])
            elif comp < 0:
                distances.append(start[i] + self.bounds)
            else:
                distances.append(float('inf'))
        
        # Use 98% of max distance to get VERY CLOSE to boundary
        max_dist = min(distances) * 0.98
        return [
            start[0] + vec[0] * max_dist,
            start[1] + vec[1] * max_dist,
            start[2]
        ]

# ==============================================================================
# 3. VALIDATION LAYER
# ==============================================================================
def is_within_bounds(pos, bounds):
    """Check if position is within ±bounds."""
    return np.all(np.abs(pos) <= bounds)

def validate_path(waypoints, target, start, bounds):
    """Comprehensive path validation."""
    if not isinstance(waypoints, list) or len(waypoints) == 0:
        return False, "Invalid plan structure: empty or not a list"
    
    # Check all positions
    all_positions = [start] + waypoints + [target]
    if not np.all(np.abs(np.array(all_positions)) <= bounds):
        invalid_points = []
        for i, pos in enumerate(all_positions):
            if not is_within_bounds(pos, bounds):
                invalid_points.append(f"Point {i}: {pos}")
        return False, f"Points exceed boundary limits: {invalid_points}"
    
    # Check waypoint progression toward target
    if waypoints:
        start_to_target = np.linalg.norm(np.array(target) - np.array(start))
        final_to_target = np.linalg.norm(np.array(waypoints[-1]) - np.array(target))
        
        if final_to_target > start_to_target * 0.1:  # Must be within 10% of target distance
            return False, f"Final waypoint {waypoints[-1]} is too far from target {target}: {final_to_target:.2f}m"
    
    return True, "Path validation successful"

# ==============================================================================
# 4. PYBULLET VISUALIZATION
# ==============================================================================
def visualize_path(start, waypoints, target, direction, bounds):
    """Draw complete path in PyBullet."""
    print("\n[System] Visualizing path...")
    
    # Clear previous debug items
    p.removeAllUserDebugItems()
    
    # Start marker (Green)
    p.addUserDebugText("START", start, [0, 1, 0], textSize=1.5)
    
    # Target marker (Red)
    p.addUserDebugText(f"GO {direction.upper()}", target, [1, 0, 0], textSize=1.5)
    
    # Compass reference
    z_h = 0.5
    p.addUserDebugText("NORTH (+X)", [40, 0, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("SOUTH (-X)", [-10, 0, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("EAST (+Y)", [0, 40, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("WEST (-Y)", [0, -40, z_h], [0, 0, 0], textSize=1.0)
    
    # Boundary walls (White box)
    bound_z = 0.1
    # Bottom edge (South)
    p.addUserDebugLine([-bounds, -bounds, bound_z], [bounds, -bounds, bound_z], [1, 1, 1], 2.0, lifeTime=0)
    # Right edge (East)
    p.addUserDebugLine([bounds, -bounds, bound_z], [bounds, bounds, bound_z], [1, 1, 1], 2.0, lifeTime=0)
    # Top edge (North)
    p.addUserDebugLine([bounds, bounds, bound_z], [-bounds, bounds, bound_z], [1, 1, 1], 2.0, lifeTime=0)
    # Left edge (West)
    p.addUserDebugLine([-bounds, bounds, bound_z], [-bounds, -bounds, bound_z], [1, 1, 1], 2.0, lifeTime=0)
    
    # Add vertical corner posts to make walls visible
    wall_height = 5.0
    corners = [(-bounds, -bounds), (bounds, -bounds), (bounds, bounds), (-bounds, bounds)]
    for corner in corners:
        p.addUserDebugLine([corner[0], corner[1], bound_z], 
                          [corner[0], corner[1], wall_height], 
                          [0.7, 0.7, 0.7], 1.5, lifeTime=0)
    
    # Path lines and waypoint markers (Blue)
    prev = start
    for i, wp in enumerate(waypoints):
        line_id = p.addUserDebugLine(prev, wp, [0, 0, 1], 3.0, lifeTime=0)
        p.addUserDebugText(f"WP{i+1}", wp, [0, 0, 1], textSize=1.0)
        prev = wp
    
    # Final approach line (Purple)
    if waypoints:
        p.addUserDebugLine(waypoints[-1], target, [0.5, 0, 0.5], 2.0, lifeTime=0)
    
    # Distance labels
    if waypoints:
        total_path_dist = np.linalg.norm(np.array(waypoints[-1]) - np.array(start))
        p.addUserDebugText(f"Path length: {total_path_dist:.1f}m", 
                          [start[0], start[1], start[2]+2], [1, 1, 0], textSize=1.2)

    print(f"[System] Plotted {len(waypoints)} waypoints")
    print(f"[System] Start: {start}")
    print(f"[System] Target: {target}")
    print(f"[System] Final waypoint: {waypoints[-1] if waypoints else 'None'}")
    print(f"[System] Final distance to target: {np.linalg.norm(np.array(waypoints[-1]) - np.array(target)):.2f}m")
    print("[System] Press Ctrl+C to exit")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def run_demo():
    """Execute story-driven navigation mission."""
    # API Key setup
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY in .env file")
        return
    
    # PyBullet setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Mission configuration
    start_pos = [0.0, 0.0, 1.0]
    bounds = 50.0
    
    # STORY-BASED MISSION (User input)
    print("Enter a natural language mission story for the drone (e.g., 'Emergency beacon detected to the north-east'):")
    mission_story = input("> ").strip()
    if not mission_story:
        mission_story = "Emergency beacon detected deep in the jungle to the north-east. Investigate immediately."
        print(f"Using default story: {mission_story}")
    
    # Initialize Navigator
    navigator = Navigator(api_key=api_key, bounds=bounds, waypoint_spacing=5.0)
    
    # Generate path from story
    print("\n" + "🚁"*35)
    print("STORY-DRIVEN DRONE NAVIGATION SYSTEM")
    print("🚁"*35)
    
    waypoints, target_pos, direction = navigator.plan_path(
        start_pos=start_pos,
        story=mission_story,
        num_waypoints=12  # SIGNIFICANTLY INCREASED to ensure reach
    )
    
    if not waypoints:
        print("\nMission Failed: No valid path generated")
        p.disconnect()
        return
    
    # Validate path
    is_valid, message = validate_path(waypoints, target_pos, start_pos, bounds)
    print(f"\n[Validation] {message}")
    
    if not is_valid:
        print("Mission Aborted: Path validation failed")
        p.disconnect()
        return
    
    # Visualize path
    visualize_path(start_pos, waypoints, target_pos, direction, bounds)
    
    # Keep simulation running
    print("\n" + "="*70)
    print("Simulation active. Close PyBullet window to exit.")
    print("="*70)
    while True:
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    run_demo()













