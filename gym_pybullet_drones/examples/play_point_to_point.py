"""Visualize a trained DQN/DDQN policy on PointToPointAviary."""

import argparse
import os
import time
from typing import Optional

import numpy as np
from stable_baselines3 import DQN

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType
from gym_pybullet_drones.utils.utils import str2bool as utils_str2bool
from gym_pybullet_drones.utils.utils import sync

DEFAULT_MODEL_PATH = "results/p2p_dqn_latest/best_model.zip"
DEFAULT_GUI = True
DEFAULT_EPISODES = 3
DEFAULT_SEED = 42


def parse_vec3(value: str) -> np.ndarray:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats, e.g., '1.0,0.0,1.0'")
    return np.array(parts, dtype=np.float32)


def run_episode(
    model: DQN,
    env: PointToPointAviary,
    episode_index: int,
    log_dir: Optional[str],
) -> None:
    logger = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(
            logging_freq_hz=int(env.CTRL_FREQ),
            num_drones=env.NUM_DRONES,
            output_folder=log_dir,
            colab=False,
        )

    obs, _ = env.reset()
    start_wall = time.time()

    for step in range(env._max_episode_steps + env.CTRL_FREQ * 2):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if env.OBS_TYPE == ObservationType.KIN and logger is not None:
            drone_state = env._getDroneStateVector(0)
            act = np.array(action, dtype=np.float32).squeeze()
            logger.log(
                drone=0,
                timestamp=step / env.CTRL_FREQ,
                state=np.hstack([
                    drone_state[0:3],
                    np.zeros(4),
                    drone_state[7:16],
                    np.atleast_1d(act),
                ]),
                control=np.zeros(12),
            )

        if env.GUI:
            sync(step, start_wall, env.CTRL_TIMESTEP)
            env.render()

        if terminated or truncated:
            break

    if logger is not None:
        logger.plot()


def main(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = DQN.load(args.model_path)

    for episode in range(args.episodes):
        seed = None if args.seed is None else args.seed + episode
        initial_xyzs = None
        if args.start_position is not None:
            initial_xyzs = np.array([args.start_position], dtype=np.float32)

        target_position = None
        if args.goal_position is not None:
            target_position = args.goal_position

        env = PointToPointAviary(
            gui=args.gui,
            initial_xyzs=initial_xyzs,
            target_position=target_position,
            randomize_start=args.randomize_start if initial_xyzs is None else False,
            randomize_target=args.randomize_target if target_position is None else False,
            min_start_target_separation=args.min_start_goal_distance,
            episode_len_sec=args.episode_seconds,
            velocity_scale=args.velocity_scale,
            max_xy=args.max_xy,
            max_z=args.max_z,
            target_tolerance=args.target_tolerance,
            use_built_in_obstacles=args.built_in_obstacles,
            seed=seed,
            record=args.record_video or args.record_frames,
        )

        log_dir = None
        if args.save_logs:
            run_folder = os.path.join(
                os.path.dirname(args.model_path),
                "playback",
                f"episode_{episode:03d}",
            )
            log_dir = run_folder

        print(f"[INFO] Episode {episode+1}/{args.episodes} | seed={seed}")
        try:
            run_episode(model, env, episode, log_dir)
        finally:
            if args.record_video and args.record_frames:
                print("[WARN] Both record_video and record_frames set; video recorder handles frames automatically.")
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a trained DQN policy on PointToPointAviary")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the saved model zip file")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of episodes to replay (default: 3)")
    parser.add_argument("--gui", type=utils_str2bool, default=DEFAULT_GUI,
                        help="Enable PyBullet GUI during playback")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base seed for episode rollouts")
    parser.add_argument("--randomize_start", type=utils_str2bool, default=True,
                        help="Sample a different start position per episode")
    parser.add_argument("--randomize_target", type=utils_str2bool, default=True,
                        help="Sample a different target position per episode")
    parser.add_argument("--start_position", type=parse_vec3, default=None,
                        help="Fixed start position 'x,y,z' (disables start randomization)")
    parser.add_argument("--goal_position", type=parse_vec3, default=None,
                        help="Fixed goal position 'x,y,z' (disables target randomization)")
    parser.add_argument("--min_start_goal_distance", type=float, default=0.3,
                        help="Minimum separation between start and goal when randomizing")
    parser.add_argument("--episode_seconds", type=float, default=12.0,
                        help="Maximum episode length in seconds")
    parser.add_argument("--velocity_scale", type=float, default=1.5,
                        help="Scale for discrete velocity commands (m/s)")
    parser.add_argument("--max_xy", type=float, default=2.0,
                        help="Horizontal workspace half-extent (meters)")
    parser.add_argument("--max_z", type=float, default=2.0,
                        help="Maximum flight altitude (meters)")
    parser.add_argument("--target_tolerance", type=float, default=0.1,
                        help="Distance threshold for considering the goal reached (meters)")
    parser.add_argument("--built_in_obstacles", type=utils_str2bool, default=True,
                        help="Enable the default PyBullet obstacle set")
    parser.add_argument("--record_video", type=utils_str2bool, default=False,
                        help="Enable PyBullet third-person MP4 recording")
    parser.add_argument("--record_frames", type=utils_str2bool, default=False,
                        help="Save onboard RGB frames per drone (PNG sequence)")
    parser.add_argument("--save_logs", type=utils_str2bool, default=False,
                        help="Store state/action logs and plots for each episode")
    args = parser.parse_args()

    main(args)
