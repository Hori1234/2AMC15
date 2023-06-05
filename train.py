# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:50:03 2023

@author: 20183067
"""
"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange
import time

# or maybe we want to keep gamma constant?
TWO_EXPERIMENTS_SIGMA = 0

try:
    from world import Environment
    from world.grid import Grid

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent
    from agents.q_learning_agent import QAgent
    from agents.ddqn import DDQNAgent
    from agents.mc_agent import MCAgent
    from agents.policy_agent import Policy_iteration
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from world import Environment

    # Add your agents here
    from agents.ddqn import DDQNAgent

    from agents.policy_agent import Policy_iteration


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument(
        "--no_gui", action="store_true", help="Disables rendering to train faster"
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to render at. Only used if " "no_gui is not set.",
    )
    p.add_argument(
        "--iter", type=int, default=100000, help="Number of iterations to go through."
    )
    p.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed value for the environment.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/"),
        help="Where to save training results.",
    )

    return p.parse_args()


def reward_function(grid: Grid, info: dict) -> float:
    """This is the default reward function.

    Args:
        grid: The grid the agent is moving on, in case that is needed by
            the reward function.
        info: The world info, in case that is needed by the reward function.

    Returns:
        A single floating point value representing the reward for a given
        action.
    """
    if info["agent_charging"][0] == True:
        return float(10)
    elif info["agent_moved"][0] == False:
        return float(-5)
    elif sum(info["dirt_cleaned"]) < 1:
        return float(-1)
    else:
        return float(5)


def main(
    no_gui: bool,
    iters: int,
    fps: int,
    out: Path,
    random_seed: int,
):
    """Main loop of the program."""

    # add two grid paths we'll use for evaluating
    grid_paths = [
        Path("grid_configs/simple1.grd"),
        Path("grid_configs/20-10-grid.grd"),
    ]

    # needed for the DDQN Agent
    batch_size = 32  # Define the batch size
    target_update_freq = 10  # Define the target update frequency
    # test for 2 different sigma values
    for sigma in [0, 0.4]:
        for grid in grid_paths:
            # Set up the environment and reset it to its initial state
            env = Environment(
                grid,
                no_gui,
                n_agents=1,
                agent_start_pos=[(1, 1)],
                sigma=0,
                target_fps=fps,
                random_seed=random_seed,
                reward_fn=reward_function,
            )
            obs, info = env.get_observation()

            # add all agents to test
            agents = [
                DDQNAgent(
                    0,
                    state_size=obs.shape()[0] * obs.shape()[1],
                    action_size=4,
                    hidden_size=32,
                    learning_rate=0.001,
                    gamma=0.99,
                    epsilon=1.0,
                )
            ]

            # Iterate through each agent for `iters` iterations
            for agent in agents:
                # Make sure that for the simple grid, we only run the one experiments
                if (sigma != TWO_EXPERIMENTS_SIGMA) and (
                    grid == Path("grid_configs/simple1.grd")
                ):
                    continue
                if type(agent).__name__ == "Policy_iteration":
                    iters = 1000
                # else:
                #     iters = 10
                fname = f"{type(agent).__name__}-sigma-{sigma}-gamma-{agent.gamma}-n_iters{iters}-time-{time.time()}"

                for i in trange(iters):
                    # Agent takes an action based on the latest observation and info
                    action = agent.take_action(obs, info)
                    old_state = info["agent_pos"][agent.agent_number]

                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])
                    new_state = info["agent_pos"][agent.agent_number]

                    if type(agent).__name__ == "DDQNAgent":
                        # check if teh algorithm converged
                        converged = agent.process_reward(
                            obs, info, reward, old_state, new_state, action, terminated
                        )
                        # perform a replay step
                        agent.replay(batch_size)
                        # Update the target network periodically
                        if i % target_update_freq == 0:
                            # print("Target Network Updated")
                            agent.update_target_network()
                        # decay the value of epsioone
                        agent.decay_epsilon(i)
                    else:
                        converged = agent.process_reward(obs, reward, info)

                    # If the agent is terminated, we reset the env.
                    if terminated:
                        obs, info, world_stats = env.reset()
                        if type(agent).__name__ == "Policy_iteration":
                            agent.dirty_tiles = []
                    # Early stopping criterion.
                    if converged:
                        break

                obs, info, world_stats = env.reset()
                print(world_stats)
                if type(agent).__name__ == "MCAgent":
                    agent.update_policy(optimal=True)
                if type(agent).__name__ == "Policy_iteration":
                    agent.dirty_tiles = []
                Environment.evaluate_agent(
                    grid,
                    [agent],
                    1000,
                    out,
                    sigma,
                    agent_start_pos=[(1, 1)],
                    custom_file_name=fname + f"-converged-{converged}-n-iters-{i}",
                )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.no_gui,
        args.iter,
        args.fps,
        args.out,
        args.random_seed,
    )
