import numpy as np
import sys
import environment as env
import random


def init_q_table(environment):
    """Initialize the Q table values to zero

    :param environment: Environment object
    :return: Initialized Q table
    """
    maze = environment.maze
    # Construct the empty Q table
    valid_states = np.where(np.isnan(maze) == False)
    valid_states = np.concatenate(valid_states).reshape((2, -1)).T
    valid_states = np.repeat(valid_states, 4, axis=0)
    repeat_actions = [0, 1, 2, 3]
    repeat_actions *= len(valid_states) // 4
    repeat_actions = np.array(repeat_actions).reshape((1, len(valid_states)))
    q_values = np.zeros((1, len(valid_states)), dtype=float)
    q_table = np.concatenate((valid_states, repeat_actions.T, q_values.T), axis=1, dtype=float)
    return q_table


def get_optimal_action(q_table, environment):
    """Get the action that results in the greatest Q value for the next state

    :param q_table: Array of the Q table
    :param environment: Environment object
    :return: Integer representing the best action
    """
    current_loc = environment.agent_location
    # Match the current state of the agent to the 4 corresponding rows of the Q table
    ref_q_table_slice = q_table[np.where((q_table[:, :2] == current_loc).all(axis=1))[0].T]
    max_action = np.argmax(ref_q_table_slice[:, 3])
    return max_action


def main():
    # Retrieve command line args
    maze_input, value_file, q_value_file, policy_file, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon = sys.argv[1:]
    # Initialize the environment
    environment = env.Environment(maze_input)
    q_table = init_q_table(environment)
    get_optimal_action(q_table, environment)
    # for episode in range(int(num_episodes)):
    #     for episode_action in range(int(max_episode_length)):
    #         # With probability 1 - epsilon, be exploitative
    #         if random.random() >= 1 - float(epsilon):

    return


if __name__ == "__main__":
    main()