from stable_baselines3 import DQN
import numpy as np
import sys
import traceback
from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.enums import ObservationType

try:
    print("Loading model...")
    sys.stdout.flush()
    model = DQN.load(r".\results\p2p_dqn_11.16.2025_12.52.11\best_model.zip")
    print(f"✓ Model policy type: {type(model.policy)}")
    print(f"✓ Model device: {model.device}")
    sys.stdout.flush()
    
    print("\nCreating environment (this may take 10-30 seconds)...")
    sys.stdout.flush()
    env = PointToPointAviary(
        obs=ObservationType.RGB, 
        gui=False, 
        randomize_start=False, 
        randomize_target=False,
        pyb_freq=240,
        ctrl_freq=24  # RGB uses 24Hz, KIN uses 30Hz
    )
    print("✓ Environment created")
    sys.stdout.flush()
    
    print("Resetting environment...")
    sys.stdout.flush()
    obs, info = env.reset()
    print("✓ Environment reset complete")
    sys.stdout.flush()
    
    print(f"\n=== Episode Start ===")
    print(f"Start position: {info['drone_position']}")
    print(f"Goal position: {info['target_position']}")
    print(f"Distance to target: {info['distance_to_target']:.4f}")
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    sys.stdout.flush()
    
    print(f"\n=== Predicting Action (deterministic) ===")
    # Handle obs shape: (batch, height, width, channels) -> (batch, channels, height, width)
    if obs.ndim == 4:
        obs_chw = np.ascontiguousarray(np.transpose(obs, (0, 3, 1, 2)))
    else:
        obs_chw = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)))
    print(f"Obs CHW shape: {obs_chw.shape}, dtype: {obs_chw.dtype}")
    sys.stdout.flush()
    action, _states = model.predict(obs_chw, deterministic=True)
    print(f"✓ Predicted action: {action}")
    sys.stdout.flush()
    
    print(f"\n=== Taking Step ===")
    sys.stdout.flush()
    obs, r, term, trunc, info = env.step(int(action))
    print(f"✓ Step complete")
    print(f"  Env _last_action: {env._last_action}")
    print(f"  Reward: {r:.4f}")
    print(f"  Distance to target: {info['distance_to_target']:.4f}")
    print(f"  Terminated: {term}, Truncated: {trunc}")
    sys.stdout.flush()
    
    print(f"\n=== Exploratory Prediction (deterministic=False) ===")
    sys.stdout.flush()
    if obs.ndim == 4:
        obs_chw = np.ascontiguousarray(np.transpose(obs, (0, 3, 1, 2)))
    else:
        obs_chw = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)))
    action_explore, _ = model.predict(obs_chw, deterministic=False)
    print(f"✓ Exploratory action: {action_explore}")
    sys.stdout.flush()
    
    env.close()
    print("\n✓ All tests passed!")
    
except KeyboardInterrupt:
    print("\n[INTERRUPTED by user]")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
