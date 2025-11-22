"""Curriculum training script for point-to-point navigation using DQN/DDQN."""

import argparse
import os
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (CallbackList, BaseCallback,
                                                EvalCallback, StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.enums import ObservationType
from gym_pybullet_drones.utils.utils import str2bool


DEFAULT_TOTAL_TIMESTEPS = int(3e5)
DEFAULT_EVAL_FREQ = 5000
DEFAULT_N_EVAL_EPISODES = 10
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_TARGET_REWARD = 150.0
DEFAULT_NUM_ENV = 4
DEFAULT_SEED = 0
DEFAULT_METRIC_SAMPLE_SIZE = 512


class DetailedMetricsCallback(BaseCallback):
    """Logs extra diagnostics such as Q-statistics, TD error, and sample reuse."""

    def __init__(self, sample_size: int = DEFAULT_METRIC_SAMPLE_SIZE, log_interval: int = 1000) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.log_interval = log_interval
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        if self.model._n_updates == 0:
            return True
        if self.num_timesteps % self.log_interval != 0:
            return True
        if self.model.replay_buffer is None or self.model.replay_buffer.size() < self.sample_size:
            return True

        infos = self.locals.get("infos", [])
        if infos:
            rewards = [float(info["episode"]["r"]) for info in infos if "episode" in info]
            lengths = [float(info["episode"]["l"]) for info in infos if "episode" in info]
            if rewards:
                self._episode_rewards.extend(rewards)
                self.logger.record("rollout/episode_reward_mean", np.mean(rewards))
                if len(rewards) > 1:
                    self.logger.record("rollout/episode_reward_std", np.std(rewards, ddof=0))
            if lengths:
                self._episode_lengths.extend(lengths)
                self.logger.record("rollout/episode_length_mean", np.mean(lengths))
                if len(lengths) > 1:
                    self.logger.record("rollout/episode_length_std", np.std(lengths, ddof=0))

        data = self.model.replay_buffer.sample(self.sample_size)
        actions = data.actions.long().squeeze(-1)

        with th.no_grad():
            q_values = self.model.q_net(data.observations)
            q_mean = q_values.mean().item()
            q_var = q_values.var(unbiased=False).item()

            current_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            next_q_values = self.model.q_net(data.next_observations)
            next_actions = next_q_values.argmax(dim=1)
            target_q_values = self.model.q_net_target(data.next_observations)
            target_selected = target_q_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

            td_target = data.rewards.squeeze(-1) + (1.0 - data.dones.squeeze(-1)) * self.model.gamma * target_selected
            td_error = current_q - td_target
            td_error_abs = td_error.abs()
            td_mse = 0.5 * td_error.pow(2).mean().item()

        env_count = self.training_env.num_envs if self.training_env is not None else 1
        reuse_ratio = (self.model._n_updates * self.model.batch_size) / max(1, self.model.num_timesteps * env_count)
        step_rewards = self.locals.get("rewards")
        if step_rewards is not None:
            mean_step_reward = float(np.asarray(step_rewards).mean())
        else:
            mean_step_reward = 0.0

        current_lr = float(self.model.lr_schedule(self.model._current_progress_remaining))

        self.logger.record("diagnostics/q_mean", q_mean)
        self.logger.record("diagnostics/q_variance", q_var)
        self.logger.record("diagnostics/td_error_mean", td_error.mean().item())
        self.logger.record("diagnostics/td_error_std", td_error.std(unbiased=False).item())
        self.logger.record("diagnostics/td_error_abs_mean", td_error_abs.mean().item())
        self.logger.record("diagnostics/td_error_abs_std", td_error_abs.std(unbiased=False).item())
        self.logger.record("diagnostics/td_loss_estimate", td_mse)
        self.logger.record("diagnostics/sample_reuse_ratio", reuse_ratio)
        self.logger.record("diagnostics/step_reward_mean", mean_step_reward)
        self.logger.record("diagnostics/learning_rate", current_lr)

        return True


def build_env(
    gui: bool,
    seed: Optional[int],
    obs: ObservationType,
    use_built_in_obstacles: bool,
    snapshot_dir: Optional[str],
    use_city_world: bool,
    city_size: int,
    velocity_scale: float,
    obstacle_density: float,
) -> PointToPointAviary:
    """Construct a fresh instance of the point-to-point environment."""

    ctrl_freq = 24 if obs == ObservationType.RGB else 30

    return PointToPointAviary(
        gui=gui,
        randomize_start=True,
        randomize_target=True,
        velocity_scale=velocity_scale,
        obs=obs,
        ctrl_freq=ctrl_freq,
        use_built_in_obstacles=use_built_in_obstacles,
        use_city_world=use_city_world,
        city_size=city_size,
        obstacle_density=obstacle_density,
        success_snapshot_dir=snapshot_dir,
        seed=seed,
    )


def train_stage(
    args: argparse.Namespace,
    stage_name: str,
    density: float,
    timesteps: int,
    model_path: Optional[str] = None,
    run_dir: str = "",
) -> str:
    """Run a single training stage."""
    print(f"\n{'='*80}")
    print(f"STARTING STAGE: {stage_name}")
    print(f"Obstacle Density: {density}")
    print(f"Timesteps: {timesteps}")
    print(f"{'='*80}\n")

    stage_dir = os.path.join(run_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    tensorboard_path = os.path.join(stage_dir, "tb")
    os.makedirs(tensorboard_path, exist_ok=True)

    obs_choice = args.obs_type.lower()
    obs_enum = ObservationType.RGB if obs_choice == "rgb" else ObservationType.KIN
    snapshot_dir = None
    if args.save_goal_snapshots and obs_choice == "rgb":
        snapshot_dir = os.path.join(stage_dir, "goal_snapshots")

    # Create training environment
    train_env = make_vec_env(
        lambda: build_env(
            gui=False,
            seed=None,
            obs=obs_enum,
            use_built_in_obstacles=args.built_in_obstacles,
            snapshot_dir=snapshot_dir,
            use_city_world=args.use_city_world,
            city_size=args.city_size,
            velocity_scale=args.velocity_scale,
            obstacle_density=density,
        ),
        n_envs=args.num_envs,
        seed=args.seed,
    )
    train_env = VecMonitor(train_env, os.path.join(stage_dir, "train_monitor"))
    if obs_choice == "rgb":
        train_env = VecTransposeImage(train_env)

    # Load model or create new
    policy_kwargs = dict(net_arch=[256, 256])
    policy_type = "CnnPolicy" if obs_choice == "rgb" else "MlpPolicy"
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = DQN.load(
            model_path,
            env=train_env,
            tensorboard_log=tensorboard_path,
            print_system_info=True,
            force_reset=False  # Keep buffer if compatible? Usually safer to reset buffer for new stage
        )
        # Reset replay buffer for new stage to avoid training on old distribution
        # model.replay_buffer.reset()
        print("Loaded model. Continuing training...")
    else:
        print("Initializing new model...")
        model = DQN(
            policy_type,
            train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_path,
            seed=args.seed,
            device=args.device,
        )

    metrics_callback = DetailedMetricsCallback(sample_size=args.metric_sample_size)
    
    new_logger = configure(stage_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=timesteps, callback=metrics_callback, progress_bar=args.progress_bar)
    
    final_path = os.path.join(stage_dir, "final_model.zip")
    model.save(final_path)
    train_env.close()
    
    return final_path


def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_dir = os.path.join(args.output_folder, f"p2p_curriculum_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # SINGLE STAGE: Medium City (50% density)
    # User requested single stage training at 0.5 density
    train_stage(
        args, 
        stage_name="single_stage_medium", 
        density=0.5, 
        timesteps=args.total_timesteps, 
        run_dir=run_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN/DDQN with Curriculum")
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, metavar="",
                        help=f"Total training timesteps across all stages (default: {DEFAULT_TOTAL_TIMESTEPS})")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, metavar="",
                        help="Directory to store logs and models (default: 'results')")
    parser.add_argument("--learning_rate", type=float, default=1e-4, metavar="",
                        help="Optimizer learning rate (default: 1e-4)")
    parser.add_argument("--buffer_size", type=int, default=100000, metavar="",
                        help="Replay buffer size (default: 100000)")
    parser.add_argument("--learning_starts", type=int, default=5000, metavar="",
                        help="Steps before training starts (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=256, metavar="",
                        help="Batch size (default: 256)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="",
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--train_freq", type=int, default=4, metavar="",
                        help="Gradient steps frequency (default: 4)")
    parser.add_argument("--gradient_steps", type=int, default=1, metavar="",
                        help="Gradient updates per training step (default: 1)")
    parser.add_argument("--target_update_interval", type=int, default=1000, metavar="",
                        help="Target network update interval (default: 1000)")
    parser.add_argument("--exploration_fraction", type=float, default=0.3, metavar="",
                        help="Exploration fraction (default: 0.3)")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, metavar="",
                        help="Final exploration epsilon (default: 0.05)")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENV, metavar="",
                        help=f"Number of parallel environments (default: {DEFAULT_NUM_ENV})")
    parser.add_argument("--metric_sample_size", type=int, default=DEFAULT_METRIC_SAMPLE_SIZE, metavar="",
                        help=f"Transitions sampled for diagnostics (default: {DEFAULT_METRIC_SAMPLE_SIZE})")
    parser.add_argument("--device", type=str, default="auto", metavar="",
                        help="Computation device (default: 'auto')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="",
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--progress_bar", type=str2bool, default=True, metavar="",
                        help="Display tqdm progress bar (default: True)")
    parser.add_argument("--obs_type", type=str, default="rgb", choices=["kin", "rgb"], metavar="",
                        help="Observation modality: 'kin' for kinematics, 'rgb' for vision (default: rgb)")
    parser.add_argument("--built_in_obstacles", type=str2bool, default=False, metavar="",
                        help="Load PyBullet's default obstacle set (default: False)")
    parser.add_argument("--use_city_world", type=str2bool, default=True, metavar="",
                        help="Use procedural City world obstacles (default: True)")
    parser.add_argument("--city_size", type=int, default=50, metavar="",
                        help="City half-extent used by City world (default: 50)")
    parser.add_argument("--velocity_scale", type=float, default=1.0, metavar="",
                        help="Maximum velocity scale (default: 1.0)")
    parser.add_argument("--save_goal_snapshots", type=str2bool, default=True, metavar="",
                        help="Capture and store RGB frames when the agent reaches the goal (default: True)")
    # Added for compatibility with manual density override if needed, though main() controls it
    parser.add_argument("--obstacle_density", type=float, default=1.0, metavar="",
                        help="Initial obstacle density (overridden by curriculum stages)")

    args = parser.parse_args()
    main(args)
