"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
import copy
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
        "--iter", type=int, default=10000, help="Number of iterations to go through."
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


def reward_function(grid: Grid, info: dict, agent_number: int) -> float:
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
    if info["agent_charging"][agent_number] == True:
        return float(10)
    elif info["agent_moved"][agent_number] == False:
        return float(-50)
    elif sum(info["dirt_cleaned"]) < 1:
        return float(-1)
    else:
        return float(5)


# BatteryRelated
def battery_reward_function(grid: Grid, info: dict, agent_number: int) -> float:
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
    if info["agent_charging"][agent_number] == True:
        # Reward if at charger after cleaning everything
        if grid.sum_dirt() == 0:
            return float(20)

        # # punished for going to charger with enough battery left
        # elif info["battery_left"][0] > 20:
        #     return float(-5)

        # # reward for going to charger with low battery
        # elif info["battery_left"][0] < 10:
        #     return float(20)

    # punish heavily for running out of battery
    elif info["battery_left"][agent_number] == 0:
        return float(-100)

    # punish for staying at the same location
    elif info["agent_moved"][agent_number] == False:
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
        Path("grid_configs/10x10_2_charge.grd"),
        #Path("grid_configs/20-10-grid.grd"),
        # Path("grid_configs/rooms-1.grd"),
        # Path("grid_configs/maze-1.grd"),
        # Path("grid_configs/walldirt-1.grd"),
        # Path("grid_configs/walldirt-2.grd"),
        # Path("grid_configs/simple1.grd"),
    ]

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        # BatteryRelated
        env = (
            EnvironmentBattery(
                grid,
                battery_size=battery_size,
                no_gui=no_gui,
                n_agents=2,
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
                n_agents=2,
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
            DeepQAgent(
                agent_number=0,
                learning_rate=0.00001,
                gamma=0.9,
                epsilon_decay=0.001,
                memory_size=100000,
                batch_size=32,
                tau=0.1,
                epsilon_stop=0.3,
                battery_size=battery_size,
            ),
            DeepQAgent(
                agent_number=1,
                learning_rate=0.00001,
                gamma=0.9,
                epsilon_decay=0.001,
                memory_size=100000,
                batch_size=32,
                tau=0.1,
                epsilon_stop=0.3,
                battery_size=battery_size,
            ),
            # QAgent(0)
        ]
        # Iterate through each agent for `iters` iterations
        # for agent in agents:
        fname = f"{type(agents[0]).__name__}-gamma-{agents[0].gamma}-n_iters{iters}-time-{time.time()}"
        #
        print("Agent is ", type(agents[0]).__name__, " gamma is ", agents[0].gamma)
        for i in trange(iters):
            for agent in agents:
                actions = [4] * len(agents)
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)
                actions[agent.agent_number] = action
                old_state = info["agent_pos"][agent.agent_number]
                old_tile_state = agent.tile_state.copy()
                # BatteryRelated
                old_battery_state = info["battery_left"][agent.agent_number]
                # The action is performed in the environment
                obs, reward, terminated, info = env.step(actions, agent.agent_number)

                new_state = info["agent_pos"][agent.agent_number]
                # print("info: ", info)
                # print("world_stats: ", env.world_stats)
                # print("reward: ", reward)

                converged = agent.process_reward(
                    obs,
                    info,
                    reward,
                    old_state,
                    new_state,
                    action,
                    old_battery_state,
                    terminated,
                )
                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                    print(f"Epsilon: {agent.eps}")
                    print("Terminated")
                if (old_tile_state != agent.tile_state) or terminated:
                    for other_agent in agents:
                        if other_agent != agent:
                            other_agent.update_agent(agent.dirty_tiles, agent.tile_state, terminated)

                # Early stopping criterion.
                if converged:
                    break

        agents[0].eps = 0
        agents[1].eps = 0
        obs, info, world_stats = env.reset()

        # BatteryRelated
        EnvironmentBattery.evaluate_agent(
            grid,
            agents,
            1000,
            out,
            sigma=0,
            agent_start_pos=None,
            custom_file_name=fname + f"-converged-{converged}-n-iters-{i}",
            battery_size=battery_size,
        ) if not no_battery else Environment.evaluate_agent(
            grid,
            agents,
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
        args.battery_size,  # BatteryRelated
        args.no_battery,  # BatteryRelated
    )
