import numpy as np
import sys
import pandas as pd
"""
    This program runs the synchronous value iteration algorithm on a maze, where the initial value at each state V(s) is initialized
    as 0. V(s) on an obstacle is not initialized. The agent receives a reward of -1 for each move. 
"""


def init_maze_values(maze_input):
    """ Initializes the maze file into an np array of floats.

    :param maze_input: Path to maze file
    :return: Array of floats
    """
    with open(maze_input, "r") as f:
        raw_maze_input = f.readlines()
    y_dim = len(raw_maze_input)
    raw_maze_input = [ord(char) for line in raw_maze_input for char in line.strip()]
    x_dim = len(raw_maze_input) // y_dim
    raw_maze_input = np.array(raw_maze_input, dtype=float)
    raw_maze_input[raw_maze_input == 46] = -1       # .
    raw_maze_input[raw_maze_input == 71] = 0        # G
    raw_maze_input[raw_maze_input == 83] = -1       # S
    raw_maze_input[raw_maze_input == 42] = np.nan   # *
    raw_maze_input = raw_maze_input.reshape((y_dim, x_dim))
    print(raw_maze_input)
    return raw_maze_input


def get_best_action(maze):
    """ Gets a list of the best actions in four directions without diagonals, within a radius of one.

    :param maze: Maze value array
    :return: Flattened array of the best actions
    """
    best_moves = []
    x_maze_bound = len(maze[0]) - 1
    y_maze_bound = len(maze) - 1
    goal_coord = []
    goal_mask = np.ma.masked_values(maze, 0)    # Goal mask is used to keep the goal best move at 0
    for i in range(y_maze_bound + 1):
        for j in range(x_maze_bound + 1):
            if maze[i][j] == 0:
                goal_coord += [i, j]
            if not np.isnan(maze[i][j]):
                if i == 0 and j != 0 and j != x_maze_bound:  # Top wall
                    max_action = np.nanmax(np.array([maze[i][j + 1], maze[i - 1][j], maze[i][j - 1]]))
                elif i != 0 and i != y_maze_bound and j == 0:  # Left wall
                    max_action = np.nanmax(np.array([maze[i + 1][j], maze[i][j + 1], maze[i - 1][j]]))
                elif i == y_maze_bound and j != 0 and j != x_maze_bound:  # Bottom wall
                    max_action = np.nanmax(np.array([maze[i - 1][j], maze[i][j + 1], maze[i][j - 1]]))
                elif i != 0 and i != y_maze_bound and j == x_maze_bound:  # Right wall
                    max_action = np.nanmax(np.array([maze[i + 1][j], maze[i - 1][j], maze[i][j - 1]]))
                elif i == 0 and j == 0:  # Top left corner
                    max_action = np.nanmax(np.array([maze[i][j + 1], maze[i + 1][j]]))
                elif i == y_maze_bound and j == x_maze_bound:  # Bottom right corner
                    max_action = np.nanmax(np.array([maze[i - 1][j], maze[i][j - 1]]))
                elif i == 0 and j == x_maze_bound:  # Top Right corner
                    max_action = np.nanmax(np.array([maze[i + 1][j], maze[i][j - 1]]))
                elif i == y_maze_bound and j == 0:  # Bottom left corner
                    max_action = np.nanmax(np.array([maze[i - 1][j], maze[i][j + 1]]))
                else:  # Everything else
                    max_action = np.nanmax(np.array([maze[i + 1][j], maze[i][j + 1], maze[i - 1][j], maze[i][j - 1]]))
            else:
                max_action = np.nan
            best_moves += [max_action]
    best_moves = np.array(best_moves).reshape(y_maze_bound + 1, x_maze_bound + 1) * ~goal_mask.mask
    return best_moves


def value_it(maze, num_epochs, discount_factor):
    maze_rounds = []
    best_actions = get_best_action(maze)
    current_maze = maze + discount_factor * best_actions
    maze_rounds.append(current_maze)
    prev_best_actions = best_actions
    original_best_actions = best_actions
    for t in range(1, num_epochs):
        best_actions = get_best_action(current_maze)
        actions_array_mask = np.equal(prev_best_actions, best_actions)
        current_maze = current_maze + (~actions_array_mask * (discount_factor ** (t + 1)) * original_best_actions)
        maze_rounds.append(current_maze)
        prev_best_actions = best_actions
    return current_maze


def main():
    maze_input, value_file, q_value_file, policy_file, num_epoch, discount_factor = sys.argv[1:]
    initialized_maze = init_maze_values(maze_input)
    value_maze = value_it(initialized_maze, int(num_epoch), float(discount_factor))
    value_maze = pd.DataFrame(value_maze)
    np.savetxt("test_output.txt", value_maze.values)
    print(value_maze)
    return


if __name__ == "__main__":
    main()