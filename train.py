"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange
import numpy as np
from world import Environment

# Add your agents here
from agents.q_sarsa_agent import QSARSA_Agent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument(
        "GRID",
        type=Path,
        nargs="+",
        help="Paths to the grid file to use. There can be more than " "one.",
    )
    p.add_argument(
        "--no_gui", action="store_true", help="Disables rendering to train faster"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Sigma value for the stochasticity of the environment.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to render at. Only used if " "no_gui is not set.",
    )
    p.add_argument(
        "--iter", type=int, default=1000, help="Number of iterations to go through."
    )
    p.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed value for the environment.",
    )
    p.add_argument("--out", type=Path, help="Where to save training results.")

    return p.parse_args()


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out: Path,
    random_seed: int,
):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        env = Environment(
            grid,
            no_gui,
            n_agents=1,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            reward_fn=QSARSA_Agent.heuristic,
        )
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [QSARSA_Agent(0, obs, use_sarsa=False, use_double_q=True)]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            episode = 1
            for _ in trange(iters):
                # Get the current position of the agent
                x, y = agent.get_current_position(info)
                # Agent takes an action based on the latest observation and info
                next_x, next_y, act = agent.take_action(obs, info)
                # The action is performed in the environment
                # and the agent receives the next observation and reward
                obs, reward, terminated, info = env.step([act])
                # The agent processes the reward and updates its internal state
                agent.process_reward(obs, reward, _, x, y, next_x, next_y, act)
                # If the agent is terminated, we reset the env.
                if terminated or (
                    np.where(obs == 4)[0].size == 0 and np.where(obs == 3)[0].size == 0
                ):
                    obs, info, world_stats = env.reset()
                    print("All dirt cleaned! and no charging station")
                    episode += 1

            obs, info, world_stats = env.reset()
            print(world_stats)

            Environment.evaluate_agent(grid, [agent], iters, out, 0.2)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.iter,
        args.fps,
        args.sigma,
        Path("results/"),
        args.random_seed,
    )
