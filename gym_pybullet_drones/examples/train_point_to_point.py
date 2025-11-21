"""Training script for point-to-point navigation using DQN/DDQN."""

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


DEFAULT_TOTAL_TIMESTEPS = int(2e5)
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
        success_snapshot_dir=snapshot_dir,
        seed=seed,
    )


def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_dir = os.path.join(args.output_folder, f"p2p_dqn_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tensorboard_path = os.path.join(run_dir, "tb")
    os.makedirs(tensorboard_path, exist_ok=True)

    obs_choice = args.obs_type.lower()
    if obs_choice not in {"kin", "rgb"}:
        raise ValueError("obs_type must be either 'kin' or 'rgb'")
    obs_enum = ObservationType.RGB if obs_choice == "rgb" else ObservationType.KIN
    snapshot_dir = None
    eval_snapshot_dir = None
    if args.save_goal_snapshots and obs_choice == "rgb":
        snapshot_dir = os.path.join(run_dir, "goal_snapshots")
        eval_snapshot_dir = os.path.join(run_dir, "goal_snapshots_eval")

    # Training environment (GUI optional when num_envs == 1)
    if args.gui and args.num_envs == 1:
        # Use a single env with GUI for visual inspection
        train_env = DummyVecEnv([
            lambda: build_env(
                gui=True,
                seed=None,
                obs=obs_enum,
                use_built_in_obstacles=args.built_in_obstacles,
                snapshot_dir=snapshot_dir,
                use_city_world=args.use_city_world,
                city_size=args.city_size,
                velocity_scale=args.velocity_scale,
            )
        ])
    else:
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
            ),
            n_envs=args.num_envs,
            seed=args.seed,
        )
    train_env = VecMonitor(train_env, os.path.join(run_dir, "train_monitor"))
    if obs_choice == "rgb":
        train_env = VecTransposeImage(train_env)

    # Evaluation environment (single instance, deterministic)
    # Skip eval env creation when GUI is on (PyBullet only allows one GUI instance)
    if args.gui:
        eval_env = None
        print("[INFO] GUI mode enabled - skipping separate eval environment creation")
        print("[INFO] Evaluation will be disabled during training with GUI")
    else:
        eval_seed = (args.seed + 123) if args.seed is not None else None

        def make_eval_env() -> Monitor:
            return Monitor(
                build_env(
                    gui=False,
                    seed=eval_seed,
                    obs=obs_enum,
                    use_built_in_obstacles=args.built_in_obstacles,
                    snapshot_dir=eval_snapshot_dir,
                    use_city_world=args.use_city_world,
                    city_size=args.city_size,
                    velocity_scale=args.velocity_scale,
                )
            )

        eval_vec_env: DummyVecEnv = DummyVecEnv([make_eval_env])
        eval_vec_env = VecMonitor(eval_vec_env, os.path.join(run_dir, "eval_monitor"))
        if obs_choice == "rgb":
            eval_vec_env = VecTransposeImage(eval_vec_env)
        eval_env = eval_vec_env

    policy_kwargs = dict(net_arch=[256, 256])
    policy_type = "CnnPolicy" if obs_choice == "rgb" else "MlpPolicy"
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
    
    # Only setup evaluation callback if eval_env exists (not in GUI mode)
    if eval_env is not None:
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=args.target_reward,
            verbose=1,
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=stop_callback,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=args.n_eval_episodes,
        )
        callback = CallbackList([eval_callback, metrics_callback])
    else:
        callback = metrics_callback

    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_model_path)

    best_model_path = os.path.join(run_dir, "best_model.zip")
    if not os.path.isfile(best_model_path) and os.path.isfile(final_model_path):
        shutil.copy(final_model_path, best_model_path)

    # Only run final evaluation if eval_env exists (not in GUI mode)
    if eval_env is not None:
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        print("=" * 80)
        print("Evaluation summary")
        print(f"Episodes: {args.n_eval_episodes}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Artifacts saved to: {run_dir}")
        print("=" * 80)
        
        eval_env.close()
    else:
        print("=" * 80)
        print("Training completed in GUI mode (evaluation skipped)")
        print(f"Artifacts saved to: {run_dir}")
        print("=" * 80)
    
    train_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN/DDQN on PointToPointAviary")
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, metavar="",
                        help=f"Total training timesteps (default: {DEFAULT_TOTAL_TIMESTEPS})")
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_EVAL_FREQ, metavar="",
                        help=f"Evaluation frequency in steps (default: {DEFAULT_EVAL_FREQ})")
    parser.add_argument("--n_eval_episodes", type=int, default=DEFAULT_N_EVAL_EPISODES, metavar="",
                        help=f"Episodes per evaluation (default: {DEFAULT_N_EVAL_EPISODES})")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, metavar="",
                        help="Directory to store logs and models (default: 'results')")
    parser.add_argument("--target_reward", type=float, default=DEFAULT_TARGET_REWARD, metavar="",
                        help=f"Reward threshold to early-stop (default: {DEFAULT_TARGET_REWARD})")
    parser.add_argument("--learning_rate", type=float, default=1e-3, metavar="",
                        help="Optimizer learning rate (default: 1e-3)")
    parser.add_argument("--buffer_size", type=int, default=200000, metavar="",
                        help="Replay buffer size (default: 200000)")
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
    parser.add_argument("--gui", type=str2bool, default=False, metavar="",
                        help="Render GUI for training (num_envs must be 1) and evaluation (default: False)")
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

    args = parser.parse_args()
    main(args)
