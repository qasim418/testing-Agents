"""Quick visualization of the PointToPointAviary layout with built-in obstacles."""

import argparse
import time
from typing import Optional

import numpy as np

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.utils import str2bool as utils_str2bool

def run_demo(duration: float,
             seed: Optional[int],
             gui: bool,
             velocity_scale: float,
             max_xy: float,
             max_z: float,
             target_tolerance: float,
             episode_seconds: float,
             randomize_start: bool,
             randomize_target: bool,
             start_position: Optional[np.ndarray],
             goal_position: Optional[np.ndarray]) -> None:
    initial_xyzs = None if start_position is None else np.array([start_position], dtype=np.float32)
    env = PointToPointAviary(
        gui=gui,
        initial_xyzs=initial_xyzs,
        target_position=goal_position,
        randomize_start=randomize_start if start_position is None else False,
        randomize_target=randomize_target if goal_position is None else False,
        min_start_target_separation=0.5,
        episode_len_sec=max(episode_seconds, 1.0),
        velocity_scale=velocity_scale,
        max_xy=max_xy,
        max_z=max_z,
        target_tolerance=target_tolerance,
        use_built_in_obstacles=True,
        seed=seed,
    )

    obs, info = env.reset()
    print("[INFO] Initial target position:", info.get("target_position"))

    action = 0  # hover command from discrete set
    sim_start = time.time()
    steps_limit = None if duration <= 0 else int(duration * env.CTRL_FREQ)
    step = 0
    try:
        while steps_limit is None or step < steps_limit:
            _, _, terminated, truncated, info = env.step(action)
            if env.GUI:
                env.render()
            step += 1
            if terminated:
                print("[INFO] Drone terminated episode (success/crash); resetting...")
                obs, info = env.reset()
                step = 0
    except KeyboardInterrupt:
        print("[INFO] Demo interrupted by user")
    finally:
        env.close()
    elapsed = time.time() - sim_start
    print(f"[INFO] Demo finished in {elapsed:.1f}s, steps executed: {step}")


def parse_vec3(text: str) -> np.ndarray:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats, e.g., '1.0,0.0,1.0'")
    return np.array(parts, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show PointToPointAviary obstacles and layout")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds to keep the simulation running (0 for until Ctrl+C)")
    parser.add_argument("--gui", type=utils_str2bool, default=True,
                        help="Whether to open the PyBullet GUI")
    parser.add_argument("--velocity_scale", type=float, default=1.5,
                        help="Scale for discrete velocity commands")
    parser.add_argument("--max_xy", type=float, default=4.0,
                        help="Horizontal workspace half-extent (meters)")
    parser.add_argument("--max_z", type=float, default=3.0,
                        help="Maximum allowed altitude")
    parser.add_argument("--target_tolerance", type=float, default=0.1,
                        help="Radius for arrival detection")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for obstacle placement")
    parser.add_argument("--episode_seconds", type=float, default=600.0,
                        help="Internal episode horizon (seconds) before truncation")
    parser.add_argument("--randomize_start", type=utils_str2bool, default=True,
                        help="Randomize start position each reset when not fixed")
    parser.add_argument("--randomize_target", type=utils_str2bool, default=True,
                        help="Randomize goal position each reset when not fixed")
    parser.add_argument("--start_position", type=parse_vec3, default=None,
                        help="Fix the drone start position 'x,y,z'")
    parser.add_argument("--goal_position", type=parse_vec3, default=None,
                        help="Fix the goal position 'x,y,z'")
    args = parser.parse_args()

    goal = None if args.goal_position is None else np.array(args.goal_position, dtype=np.float32)
    start = None if args.start_position is None else np.array(args.start_position, dtype=np.float32)

    run_demo(
        duration=args.duration,
        seed=args.seed,
        gui=args.gui,
        velocity_scale=args.velocity_scale,
        max_xy=args.max_xy,
        max_z=args.max_z,
        target_tolerance=args.target_tolerance,
        episode_seconds=args.episode_seconds,
        randomize_start=args.randomize_start,
        randomize_target=args.randomize_target,
        start_position=start,
        goal_position=goal,
    )


if __name__ == "__main__":
    main()
