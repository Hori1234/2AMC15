# heuristic_distance_to_dirt
from world.grid import Grid
import numpy as np


def heuristic_distance_to_dirt(grid: Grid, info: None | dict) -> float:
    """Count the number of tiles cleand by the agent.

    Args:
        grid: The grid object.
        info: The info dictionary returned by the environment.

    Returns:
        The number of tiles cleaned by the agent.
    """
    obs = grid.cells

    def calculate_distance(current_state, dirt_pos):
        x1, y1 = current_state
        x2, y2 = dirt_pos
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance

    # Calculate the distance to the nearest dirt tile, avoiding obstacles
    current_state = info["agent_pos"][0]
    empty_positions = [
        (x, y) for x, y in zip(np.where(obs == 0)[0], np.where(obs == 0)[1])
    ]
    dirt_positions = [
        (x, y) for x, y in zip(np.where(obs == 3)[0], np.where(obs == 3)[1])
    ]
    walls_positions = [
        (x, y) for x, y in zip(np.where(obs == 2)[0], np.where(obs == 2)[1])
    ]

    if len(dirt_positions) == 0:
        charging_station_positions = (
            np.where(obs == 4)[0][0],
            np.where(obs == 4)[1][0],
        )

        # No dirt tiles remaining, return a high reward
        dirt_positions.append(charging_station_positions)

    min_distance = float("inf")
    for dirt_pos in dirt_positions:
        distance = calculate_distance(current_state, dirt_pos)
        min_distance = min(min_distance, distance)

    # Adjust the distance by the number of obstacles along the path
    path_obstacles = set(walls_positions) - set() - set(dirt_positions)
    num_obstacles = len(path_obstacles)
    adjusted_distance = min_distance + num_obstacles

    return 1.0 / adjusted_distance if adjusted_distance != float("inf") else 0.0


def heuristic_distance_to_dirt_with_trajectory(grid: Grid, info: None | dict) -> float:
    """Count the number of tiles cleaned by the agent.

    Args:
        grid: The grid object.
        info: The info dictionary returned by the environment.

    Returns:
        The number of tiles cleaned by the agent.
    """

    # Calculate the distance to the nearest dirt tile, avoiding obstacles
    obs = grid.cells

    """ Here is the explanation for the code above:
        1. We use Manhattan distance to calculate the distance between two points.
        2. The distance is the sum of the absolute values of the differences of the coordinates.
        """

    def calculate_distance(current_state, dirt_pos):
        x1, y1 = current_state
        x2, y2 = dirt_pos
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance

    """ Here is the explanation for the code above:
        1. The "walls" variable is a set of tuples, where each tuple is the (x, y) coordinate of a wall in the maze.
        2. The "empty_cells" variable is also a set of tuples, where each tuple is the (x, y) coordinate of an empty cell in the maze.
        3. The "state" variable is a tuple, containing the (x, y) coordinate of the current state.
        4. The "is_room_exit" function should return True if the current state is next to a wall or is not next to an empty cell. Otherwise, it should return False. 
        """

    def is_room_exit(state, empty_cells, walls):
        x, y = state
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if (next_x, next_y) not in empty_cells or (next_x, next_y) in walls:
                return True

        return False

    """ The code above does the following:
        1. Find all the room exit tiles.
        2. Calculate the distances from the current state to each room exit.
        3. Find the closest room exit and add it to the trajectory. 
        """

    def trajectory_calculation(state, empty_cells, walls):
        trajectory = []
        room_exits = []
        # Find the room exit tiles
        for empty_cell in empty_cells:
            if is_room_exit(empty_cell, empty_cells, walls):
                room_exits.append(empty_cell)

        if len(room_exits) == 0:
            # No room exits found, return an empty trajectory
            return trajectory

        # Calculate the distances from the current state to each room exit
        distances_to_exits = [
            calculate_distance(state, exit_tile) for exit_tile in room_exits
        ]

        # Find the closest room exit and add it to the trajectory
        closest_exit_index = np.argmin(distances_to_exits)
        closest_exit = room_exits[closest_exit_index]
        trajectory.append(closest_exit)

        return trajectory

    # Calculate the distance to the nearest dirt tile, avoiding obstacles
    current_state = info["agent_pos"][0]
    empty_positions = [
        (x, y) for x, y in zip(np.where(obs == 0)[0], np.where(obs == 0)[1])
    ]
    dirt_positions = [
        (x, y) for x, y in zip(np.where(obs == 3)[0], np.where(obs == 3)[1])
    ]
    walls_positions = [
        (x, y) for x, y in zip(np.where(obs == 2)[0], np.where(obs == 2)[1])
    ]

    if len(dirt_positions) == 0:
        charging_station_positions = (
            np.where(obs == 4)[0][0],
            np.where(obs == 4)[1][0],
        )

        # No dirt tiles remaining, return a high reward
        dirt_positions.append(charging_station_positions)

    min_distance = float("inf")
    for dirt_pos in dirt_positions:
        distance = calculate_distance(current_state, dirt_pos)
        min_distance = min(min_distance, distance)

    # Adjust the distance by the number of obstacles along the path
    path_obstacles = set(walls_positions) - set(dirt_positions)
    num_obstacles = len(path_obstacles)
    adjusted_distance = min_distance + num_obstacles

    # Calculate the trajectory to the dirt position including the room exit
    trajectory = trajectory_calculation(current_state, empty_positions, walls_positions)

    # Calculate the balance reward based on trajectory
    balance_reward = 0.0
    for tile in trajectory:
        if tile == current_state:
            balance_reward += 1.0
        else:
            distance_to_tile = calculate_distance(current_state, tile)
            balance_reward += 1.0 / distance_to_tile

    total_reward = 1.0 / adjusted_distance + balance_reward
    return total_reward if total_reward != float("inf") else 0.0
