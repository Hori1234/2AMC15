"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange

from world import Environment

# Add your agents here
from agents.q_learn_agent import QLearn_Agent
from agents.mc_agent import MCAgent

from world.grid import Grid


# from environment import reward_function


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


def custom_reward_function(grid: Grid, info: dict) -> float:
    """
    Reward of taking each step: -1 (to encourage the agent to clean
    faster and in case of an action that takes the agent to an obstacle,
    the agent remains in the same tile but gets a penalty for wasting an action)

    Reward of staying in the same tile: -5 (to encourage the agent to keep moving,
    because staying in the same location is not helping anyone)

    Reward of cleaning each dirty tile (orange): +5 (you do not need to put too
    many dirty tiles in the grid, for instance in a 10x10 grid, 2-3 dirty tiles
    would be enough).

    Reward of reaching the charging station after cleaning all the dirty tiles: +10
    (if the agent reaches the charging station but there are still some dirty tiles
    left, the reward would be -1, similar to a step reward)
    """
    # TODO: Als er meerdere agents tegelijk zijn dan gaat dit mis
    # want dan zal dirt_cleaned een lijst zijn met de dirt_cleaned van alle agents
    # dus bijv [0, 1, 0] en moet je dus niet altijd de 1e index hebben zoals ik dat nu wel heb gedaan
    if info["dirt_cleaned"][0] > 0:
        reward = 5
    elif not info["agent_moved"][0]:
        reward = -5
    elif info["agent_charging"][0]:
        if grid.sum_dirt() == 0:
            reward = 10
        else:
            reward = -1
    else:
        reward = -1

    return reward


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
            # agent_start_pos=[(5, 10)],
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            reward_fn=custom_reward_function,
        )
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            # QLearn_Agent(0,obs),
            MCAgent(0, obs)
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            for _ in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])

                # If the agent is terminated, we reset the env.
                # for the 3 dirt env, also terminate if no drit left
                # note that this extra statement has to be removed with new envs
                if terminated or env.grid.sum_dirt() == 0:
                    obs, info, world_stats = env.reset()
                    agent.dirt_found = [0, 0, 0]

                agent.process_reward(obs, reward, info)
            obs, info, world_stats = env.reset()
            agent.update_policy(optimal=True)
            print(world_stats)
            Environment.evaluate_agent(
                grid, [agent], 100, out, 0.2, agent_start_pos=[(7, 5)]
            )


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
