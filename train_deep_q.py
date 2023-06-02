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

# This value of sigma will be used for both grids
# The other sigma
TWO_EXPERIMENTS_SIGMA = 0

try:
    from world import Environment
    from world.grid import Grid

    # Add your agents here
    from agents.deep_q_agent import DeepQAgent
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
        return float(-50)
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
        Path("grid_configs/20-10-grid.grd"),
        Path("grid_configs/rooms-1.grd"),
        Path("grid_configs/maze-1.grd"),
        Path("grid_configs/walldirt-1.grd"),
        Path("grid_configs/walldirt-2.grd"),
    ]

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        env = Environment(
            grid,
            no_gui,
            n_agents=1,
            agent_start_pos=None,
            target_fps=fps,
            sigma=0,
            random_seed=random_seed,
            reward_fn=reward_function,
        )
        obs, info = env.get_observation()

        # add all agents to test
        agents = [
            DeepQAgent(agent_number=0, learning_rate=0.01, gamma=0.8, epsilon_decay=0.01, memory_size=1000, batch_size=500, tau=0.01),
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:

            fname = f"{type(agent).__name__}-gamma-{agent.gamma}-n_iters{iters}-time-{time.time()}"

            print("Agent is ", type(agent).__name__, " gamma is ", agent.gamma)

            for i in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)
                old_state = info["agent_pos"][agent.agent_number]

                

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])
                new_state = info["agent_pos"][agent.agent_number]

                converged = agent.process_reward(
                    obs, info, reward, old_state, new_state, action
                )

                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                    print(f'Epsilon: {agent.eps}')

                # Early stopping criterion.
                if converged:
                    break

            agent.eps = 0
            obs, info, world_stats = env.reset()
            print(world_stats)

            Environment.evaluate_agent(
                grid,
                [agent],
                1000,
                out,
                sigma=0,
                agent_start_pos=None,
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