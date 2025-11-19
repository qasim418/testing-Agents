"""
City World Viewer
Usage:
  python -m gym_pybullet_drones.examples.show_city_world --size 50 --seed 42
"""

import argparse
import time

import pybullet as p
import pybullet_data

from gym_pybullet_drones.envs.city_world import CityGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Render the procedural City world in PyBullet GUI")
    parser.add_argument("--size", type=int, default=50, help="City half-extent in meters (grid scale)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic layout")
    parser.add_argument("--fps", type=int, default=60, help="Simulation steps per second")
    return parser.parse_args()


def main():
    args = parse_args()

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    CityGenerator(client_id=cid, seed=args.seed).generate(size=int(args.size))

    # Nice overhead camera
    p.resetDebugVisualizerCamera(
        cameraDistance=60,
        cameraYaw=45,
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=cid,
    )

    dt = 1.0 / max(1, int(args.fps))
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(physicsClientId=cid)


if __name__ == "__main__":
    main()
