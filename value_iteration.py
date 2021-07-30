import numpy as np
import sys
import pandas as pd
"""
    This program runs the synchronous value iteration algorithm on a maze, where the initial value at each state V(s) is initialized
    as 0. V(s) on an obstacle is not initialized. The agent receives a reward of -1 for each move. 
"""


def init_maze_values(maze_input):
    """ Initializes the maze file into an np array of floats. Fixed for use in q-learning for now.

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
    raw_maze_input[raw_maze_input == 83] = -2       # S     # Change to -1 when doing value iteration
    raw_maze_input[raw_maze_input == 42] = np.nan   # *
    raw_maze_input = raw_maze_input.reshape((y_dim, x_dim))
    return raw_maze_input


def get_best_action(maze):
    """ Gets a list of the best actions in four directions without diagonals, within a radius of one. Places the max action value back inside
        the reference index. Therefore, the returned array is the same shape as the input maze.

    :param maze: Maze value array
    :return: Array of the best actions
    """
    best_moves = []
    x_maze_bound = len(maze[0]) - 1
    y_maze_bound = len(maze) - 1
    goal_coord = []
    goal_mask = np.ma.masked_values(maze, 0)    # Goal mask is used to keep the goal best move at 0
    for i in range(y_maze_bound + 1):           # Would love to get rid of these for loops, but they may be unavoidable...
        for j in range(x_maze_bound + 1):       # ...Maybe some sort of convolution/mask could work?
            if maze[i][j] == 0:
                goal_coord += [i, j]
            # Check all edge cases
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


def directional_sweep(maze, discount_factor):
    """ Same as get_best_action(), but does not calculate the max of the directions within a radius of 1. Instead, the potential
        actions are saved. Intuitively, this function saves all actions a at all states s, filling the sets A and S for Q value
        population. This is basically repeat code, but this is necessary in this assignment.

        Note: Actions vectors are mapped as such:  [left, up, right, down]

        :param maze: Maze value array
        :return: Array of the best actions
        """
    set_of_actions = []
    y_maze_bound = len(maze) - 1
    x_maze_bound = len(maze[0]) - 1
    goal_index = 0
    print(maze)
    for i in range(y_maze_bound + 1):
        for j in range(x_maze_bound + 1):
            if maze[i][j] == 0:
                action = np.zeros(4)
                goal_index = i + j
            # Check all edge cases
            elif not np.isnan(maze[i][j]):
                # Base action is: action = np.array([maze[i][j - 1], maze[i + 1][j], maze[i][j + 1], maze[i - 1][j]])
                if i == 0 and j != 0 and j != x_maze_bound:  # Top wall
                    action = np.array(maze[i][j - 1], [maze[i][j], maze[i][j + 1], maze[i + 1][j]])
                elif i != 0 and i != y_maze_bound and j == 0:  # Left wall
                    action = np.array([maze[i][j], maze[i - 1][j], maze[i][j + 1], maze[i + 1][j]])
                elif i == y_maze_bound and j != 0 and j != x_maze_bound:  # Bottom wall
                    action = np.array([maze[i][j - 1], maze[i - 1][j], maze[i][j + 1], maze[i][j]])
                elif i != 0 and i != y_maze_bound and j == x_maze_bound:  # Right wall
                    action = np.array([maze[i][j - 1], maze[i - 1][j], maze[i][j], maze[i + 1][j]])
                elif i == 0 and j == 0:  # Top left corner
                    action = np.array([maze[i][j], maze[i][j], maze[i][j + 1], maze[i + 1][j]])
                elif i == y_maze_bound and j == x_maze_bound:  # Bottom right corner
                    action = np.array([maze[i][j - 1], maze[i - 1][j], maze[i][j], maze[i][j]])
                elif i == 0 and j == x_maze_bound:  # Top Right corner
                    action = np.array([maze[i][j - 1], maze[i][j], maze[i][j], maze[i + 1][j]])
                elif i == y_maze_bound and j == 0:  # Bottom left corner
                    action = np.array([maze[i][j], maze[i - 1][j], maze[i][j + 1], maze[i][j]])
                else:  # Everything else
                    action = np.array([maze[i][j - 1], maze[i - 1][j], maze[i][j + 1], maze[i + 1][j]])
            else:
                action = np.empty(4)
                action.fill(np.nan)
            # Assemble mask which is added to the maze once computed
            action_mask = np.where(action == 0, -1, action * discount_factor)
            action_mask = np.where(action_mask != -1, np.fmod(action_mask, 1), action_mask)
            set_of_actions += [action + action_mask]
    set_of_actions[goal_index] = np.zeros(len(set_of_actions))
    set_of_actions = np.array([set_of_actions])
    return set_of_actions


def value_it(maze, num_epochs, discount_factor):
    """ Calculates array containing values resulting from the value iteration algorithm

    :param maze: Initialized value maze
    :param num_epochs: Number of epochs to continue value iteration
    :param discount_factor: Discount factor for each iteration step
    :return: Maze of values
    """
    maze_rounds = []
    # First epoch
    best_actions = get_best_action(maze)
    current_maze = maze + discount_factor * best_actions
    maze_rounds.append(current_maze)
    prev_best_actions = best_actions        # This changes with epoch, and allows comparison for creating mask
    original_best_actions = best_actions    # This doesn't change with epoch, is used to keep value addition constant
    for t in range(1, num_epochs):
        best_actions = get_best_action(current_maze)
        actions_array_mask = np.equal(prev_best_actions, best_actions)  # Create mask which maps the values to be updated
        current_maze = current_maze + (~actions_array_mask * (discount_factor ** (t + 1)) * original_best_actions)  # Discount factor compounds w/ time
        maze_rounds.append(current_maze)
        prev_best_actions = best_actions
    return current_maze


def q_values(maze, discount_factor):
    """
        Note: Actions are mapped as such {up=0, left=1, right=2, down=3}

    :param maze:
    :param discount_factor:
    :return:
    """
    actions = np.arange(4)

    return


def values_to_txt(filename, value_maze):
    """ Converts the value maze into a text file, of the format <i j value>

    :param filename: Path to file
    :param value_maze: Maze value array
    :return: Nothing
    """
    real_values = np.where(~np.isnan(value_maze))
    flat_value_maze = value_maze.flatten()[np.where(~np.isnan(value_maze.flatten()))[0]]
    index_array = np.array([real_values[0], real_values[1]])
    x = flat_value_maze.shape[0]
    f_string_array = np.empty((3, x))
    f_string_array[0:] = [index_array[0], index_array[1], flat_value_maze]
    np.savetxt(filename, f_string_array.T, fmt="%s")


def main():
    # Receive input/output args and hyper-parameters
    maze_input, value_file, q_value_file, policy_file, num_epoch, discount_factor = sys.argv[1:]
    # Initialize the maze be replacing symbols with numbers
    initialized_maze = init_maze_values(maze_input)
    # Do value iteration
    value_maze = value_it(initialized_maze, int(num_epoch), float(discount_factor))
    # Save the values to a txt file
    values_to_txt(value_file, value_maze)
    set_of_actions = directional_sweep(value_maze, float(discount_factor))
    value_maze = pd.DataFrame(value_maze)
    print(value_maze)
    print(set_of_actions)
    return


if __name__ == "__main__":
    main()