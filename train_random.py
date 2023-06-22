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

try:
    from world import Environment
    from world.grid import Grid

    # BatteryRelated
    from world import EnvironmentBattery

    # Add your agents here
    from agents.deep_q_agent import DeepQAgent
    from agents.random_agent import RandomAgent

    from agents.q_learning_agent import QAgent
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

    # BatteryRelated
    from world import EnvironmentBattery

    # Add your agents here
    from agents.deep_q_agent import DeepQAgent
    from agents.random_agent import RandomAgent


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

    # BatteryRelated
    p.add_argument(
        "--battery_size",
        type=int,
        default=1000,
        help="Number of actions the agent can take before it needs to recharge.",
    )

    # BatteryRelated
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


# BatteryRelated
def battery_reward_function(grid: Grid, info: dict) -> float:
    """
    Custom reward function used in the Battery Environment.

    The agent is punished most if he runs out of battery. If the agent
    goes to the charger, this is rewarded if the agent has cleaned all
    dirt or if the agent has low battery. If the agent goes to the charger
    with enough battery left, without having cleaned all dirt, this is punished.

    Furthermore, staying at the same location is punished and moving without
    cleaning is punished a little. Cleaning dirt is rewarded.
    """
    # Agent at charger
    if info["agent_charging"][0] == True:
        # Reward if at charger after cleaning everything
        if grid.sum_dirt() == 0:
            return float(100)

    # punish heavily for running out of battery
    elif info["battery_left"][0] == 0:
        return float(-50)

    # punish for staying at the same location
    elif info["agent_moved"][0] == False:
        return float(-50)

    # punish a little for moving without cleaning
    elif sum(info["dirt_cleaned"]) < 1:
        return float(-1)

    # reward for cleaning dirt
    else:
        return float(5)


def main(
    no_gui: bool,
    iters: int,
    fps: int,
    out: Path,
    random_seed: int,
    battery_size: int,  # BatteryRelated
    no_battery: bool,  # BatteryRelated
):
    """Main loop of the program."""

    # add two grid paths we'll use for evaluating
    grid_paths = [
        # Path("grid_configs/single-agent-map.grd"),
        # Path("grid_configs/20-10-grid.grd"),
        # Path("grid_configs/rooms-1.grd"),
        # Path("grid_configs/maze-1.grd"),
        # Path("grid_configs/walldirt-1.grd"),
        # Path("grid_configs/walldirt-2.grd"),
        Path("grid_configs/simple1.grd"),
    ]

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        # BatteryRelated
        env = (
            EnvironmentBattery(
                grid,
                battery_size=battery_size,
                no_gui=no_gui,
                n_agents=1,
                agent_start_pos=None,
                target_fps=fps,
                sigma=0,
                random_seed=random_seed,
                reward_fn=battery_reward_function,
            )
            if not no_battery
            else Environment(
                grid,
                no_gui=no_gui,
                n_agents=1,
                agent_start_pos=None,
                target_fps=fps,
                sigma=0,
                random_seed=random_seed,
                reward_fn=reward_function,
            )
        )
        obs, info = env.get_observation()

        # add all agents to test
        agents = [
            RandomAgent(0)
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            # fname = f"{type(agent).__name__}-gamma-{agent.gamma}-n_iters{iters}-time-{time.time()}"
            fname = f"{type(agent).__name__}-n_iters{iters}-time-{time.time()}"
            print("Agent is ", type(agent).__name__)

            # Define a variable to accumulate the total loss
            total_loss = []

            for i in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)
                old_state = info["agent_pos"][agent.agent_number]

                # BatteryRelated
                old_battery_state = info["battery_left"][agent.agent_number]
                # print(old_battery_state)

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])

                # converged = agent.process_reward(
                #     obs,
                #     info,
                #     reward,
                #     old_state,
                #     action,
                #     old_battery_state,
                # )

                # total_loss += [agent.loss]


                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                    # print(f"Epsilon: {agent.eps}")
                    # print("Terminated")
                    # # Compute the average loss
                    # average_loss = sum(total_loss) / len(total_loss)

                    # # Print the average loss
                    # print("Average Loss:", average_loss)

                    total_loss = []

                # # Early stopping criterion.
                # if converged:
                #     break

            agent.eps = 0
            obs, info, world_stats = env.reset()

            # BatteryRelated
            EnvironmentBattery.evaluate_agent(
                grid,
                [agent],
                1000,
                out,
                sigma=0,
                agent_start_pos=None,
                custom_file_name=fname + f"-n-iters-{i}-sigma-0.0",
                battery_size=battery_size,
            ) if not no_battery else Environment.evaluate_agent(
                grid,
                [agent],
                1000,
                out,
                sigma=0,
                agent_start_pos=None,
                custom_file_name=fname + f"-n-iters-{i}",
            )

            EnvironmentBattery.evaluate_agent(
                grid,
                [agent],
                1000,
                out,
                sigma=0.2,
                agent_start_pos=None,
                custom_file_name=fname + f"-n-iters-{i}-sigma-0.2",
                battery_size=battery_size,
            ) if not no_battery else Environment.evaluate_agent(
                grid,
                [agent],
                1000,
                out,
                sigma=0,
                agent_start_pos=None,
                custom_file_name=fname + f"-n-iters-{i}",
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.no_gui,
        args.iter,
        args.fps,
        args.out,
        args.random_seed,
        args.battery_size,  # BatteryRelated
        args.no_battery,  # BatteryRelated
    )
