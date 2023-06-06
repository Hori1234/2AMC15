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
import numpy as np

# This value of sigma will be used for both grids
# The other sigma
TWO_EXPERIMENTS_SIGMA = 0

try:
    from world import EnvironmentBattery
    from world import Environment
    from world.grid import Grid

    # Add your agents here
    from agents.q_learning_agent import QAgent
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

    from world import EnvironmentBattery
    from world import Environment

    # Add your agents here
    from agents.policy_agent import Policy_iteration
    from agents.q_learning_agent import QAgent
    from agents.mc_agent import MCAgent


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

    p.add_argument(
        "--battery_size",
        type=int,
        default=1000,
        help="Number of actions the agent can take before it needs to recharge.",
    )

    p.add_argument(
        "--no_battery", action="store_true", help="Disables the battery feature."
    )

    return p.parse_args()


def reward_function(grid: Grid, info: dict) -> float:
    """Custom reward function. Punish the agent most for staying at the same
    location. Punish a bit less for moving without cleaning. Reward for cleaning
    dirt and reward the most for going back to the charging station when all dirt
    is gone.

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
    battery_size: int,
    no_battery: bool,
):
    """Main loop of the program."""

    # add two grid paths we'll use for evaluating
    grid_paths = [
        Path("grid_configs/simple1.grd"),
        Path("grid_configs/20-10-grid.grd"),
    ]

    # test for 2 different sigma values
    for sigma in [0, 0.4]:
        for grid in grid_paths:
            # Set up the environment and reset it to its initial state
            env = (
                EnvironmentBattery(
                    grid,
                    battery_size=battery_size,
                    no_gui=no_gui,
                    n_agents=1,
                    agent_start_pos=[(1, 1)],
                    target_fps=fps,
                    sigma=0,
                    random_seed=random_seed,
                    reward_fn=reward_function,
                )
                if not no_battery
                else Environment(
                    grid,
                    no_gui=no_gui,
                    n_agents=1,
                    agent_start_pos=[(1, 1)],
                    target_fps=fps,
                    sigma=0,
                    random_seed=random_seed,
                    reward_fn=reward_function,
                )
            )

            obs, info = env.get_observation()

            charger_loc = np.where(obs == 4)
            cx, cy = charger_loc[0][0], charger_loc[1][0]

            # add all agents to test
            agents = [
                QAgent(0, learning_rate=1, gamma=0.6, epsilon_decay=0.001),
                # QAgent(0, learning_rate=1, gamma=0.9, epsilon_decay=0.001),
                #     MCAgent(
                #         0,
                #         obs,
                #         gamma=0.6,
                #         epsilon=0.1,
                #         len_episode=100,
                #         n_times_no_policy_change_for_convergence=100,
                #     ),
                #     MCAgent(
                #         0,
                #         obs,
                #         gamma=0.9,
                #         epsilon=0.1,
                #         len_episode=100,
                #         n_times_no_policy_change_for_convergence=100,
                #     ),
                #     Policy_iteration(
                #         0,
                #         gamma=0.6,
                #     ),
                #     Policy_iteration(
                #         0,
                #         gamma=0.9,
                #     ),
            ]

            # Iterate through each agent for `iters` iterations
            for agent in agents:
                # Make sure that for the simple grid, we only run the one experiments
                if (sigma != TWO_EXPERIMENTS_SIGMA) and (
                    grid == Path("grid_configs/simple1.grd")
                ):
                    continue

                fname = f"{type(agent).__name__}-sigma-{sigma}-gamma-{agent.gamma}-n_iters{iters}-time-{time.time()}"

                print("Agent is ", type(agent).__name__, " gamma is ", agent.gamma)

                for i in trange(iters):
                    # Agent takes an action based on the latest observation and info
                    action = agent.take_action(obs, info)
                    old_state = info["agent_pos"][agent.agent_number]

                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])
                    new_state = info["agent_pos"][agent.agent_number]

                    if type(agent).__name__ == "QAgent":
                        converged = agent.process_reward(
                            obs, info, reward, old_state, new_state, action
                        )

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

                    if type(agent).__name__ == "Policy_iteration" and i == 1000:
                        break

                obs, info, world_stats = env.reset()

                if type(agent).__name__ == "MCAgent":
                    agent.update_policy(optimal=True)

                if type(agent).__name__ == "Policy_iteration":
                    agent.dirty_tiles = []

                EnvironmentBattery.evaluate_agent(
                    grid,
                    [agent],
                    1000,
                    out,
                    sigma,
                    agent_start_pos=[(1, 1)],
                    custom_file_name=fname + f"-converged-{converged}-n-iters-{i}",
                    battery_size=battery_size,
                ) if not no_battery else Environment.evaluate_agent(
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
        args.battery_size,
        args.no_battery,
    )
