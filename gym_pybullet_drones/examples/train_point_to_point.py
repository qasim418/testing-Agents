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

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
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

    def _on_step(self) -> bool:
        if self.model._n_updates == 0:
            return True
        if self.num_timesteps % self.log_interval != 0:
            return True
        if self.model.replay_buffer is None or self.model.replay_buffer.size() < self.sample_size:
            return True

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

        env_count = self.training_env.num_envs if self.training_env is not None else 1
        reuse_ratio = (self.model._n_updates * self.model.batch_size) / max(1, self.model.num_timesteps * env_count)
        step_rewards = self.locals.get("rewards")
        if step_rewards is not None:
            mean_step_reward = float(np.asarray(step_rewards).mean())
        else:
            mean_step_reward = 0.0

        self.logger.record("diagnostics/q_mean", q_mean)
        self.logger.record("diagnostics/q_variance", q_var)
        self.logger.record("diagnostics/td_error_mean", td_error.mean().item())
        self.logger.record("diagnostics/td_error_std", td_error.std(unbiased=False).item())
        self.logger.record("diagnostics/td_error_abs_mean", td_error_abs.mean().item())
        self.logger.record("diagnostics/td_error_abs_std", td_error_abs.std(unbiased=False).item())
        self.logger.record("diagnostics/sample_reuse_ratio", reuse_ratio)
        self.logger.record("diagnostics/step_reward_mean", mean_step_reward)

        return True


def build_env(gui: bool, seed: Optional[int]) -> PointToPointAviary:
    """Helper to construct a fresh instance of the point-to-point environment."""
    return PointToPointAviary(
        gui=gui,
        randomize_start=True,
        randomize_target=True,
        velocity_scale=1.5,
        seed=seed,
    )


def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_dir = os.path.join(args.output_folder, f"p2p_dqn_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tensorboard_path = os.path.join(run_dir, "tb")
    os.makedirs(tensorboard_path, exist_ok=True)

    # Vectorized training environment
    train_env = make_vec_env(
        lambda: build_env(gui=False, seed=args.seed),
        n_envs=args.num_envs,
        seed=args.seed,
    )

    # Evaluation environment (single instance, deterministic)
    eval_env = Monitor(build_env(gui=args.gui_eval, seed=args.seed + 123 if args.seed is not None else None))

    policy_kwargs = dict(net_arch=[256, 256])
    model = DQN(
        "MlpPolicy",
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
        render=args.gui_eval,
        n_eval_episodes=args.n_eval_episodes,
    )

    metrics_callback = DetailedMetricsCallback(sample_size=args.metric_sample_size)
    callback = CallbackList([eval_callback, metrics_callback])

    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_model_path)

    best_model_path = os.path.join(run_dir, "best_model.zip")
    if not os.path.isfile(best_model_path) and os.path.isfile(final_model_path):
        shutil.copy(final_model_path, best_model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=args.gui_eval,
    )

    print("=" * 80)
    print("Evaluation summary")
    print(f"Episodes: {args.n_eval_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Artifacts saved to: {run_dir}")
    print("=" * 80)

    eval_env.close()
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
    parser.add_argument("--gui_eval", type=str2bool, default=False, metavar="",
                        help="Render evaluation episodes (default: False)")
    parser.add_argument("--progress_bar", type=str2bool, default=True, metavar="",
                        help="Display tqdm progress bar (default: True)")

    args = parser.parse_args()
    main(args)
