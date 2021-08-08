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

    # Enumerate all possible actions per state
    repeat_actions = [0, 1, 2, 3]
    repeat_actions *= len(valid_states) // 4
    repeat_actions = np.array(repeat_actions).reshape((1, len(valid_states)))

    # Initialize Q values to 0
    q_values = np.zeros((1, len(valid_states)), dtype=float)

    # Combine to create fully-initialized Q-table
    q_table = np.concatenate((valid_states, repeat_actions.T, q_values.T), axis=1, dtype=float)
    return q_table


def get_optimal_action(q_table, agent_state):
    """Get the action that results in the greatest Q value for the next state

    :param q_table: Array containing q values
    :param agent_state: Current state of the agent
    :return: Integer representing the best action
    """
    # Match the current state of the agent to the 4 corresponding rows of the Q table
    ref_q_table_slice = q_table[np.where((q_table[:, :2] == agent_state).all(axis=1))[0].T]
    max_action = np.argmax(ref_q_table_slice[:, 3])
    return max_action


def update_q_table(q_table, state, action, reward, next_state, learning_rate, discount_factor):
    """Updates the given entry in the Q table

    :param q_table: Array containing q values
    :param state: Current state coordinates
    :param action: Current action int
    :param reward: Immediate reward int
    :param next_state: Next state coordinates
    :param learning_rate: Learning rate float
    :param discount_factor: Discount factor float
    :return: None
    """
    # Get the working index within the Q-table
    q_tuple = np.concatenate((state, action), axis=None)
    q_table_row_index = np.where((q_table[:, :3] == q_tuple).all(axis=1))

    # Get future expected reward
    max_next_action = get_optimal_action(q_table, next_state)
    future_reward_tuple = np.concatenate((next_state, max_next_action), axis=None)
    future_reward_row_index = np.where((q_table[:, :3] == future_reward_tuple).all(axis=1))

    # Calculate Q-value
    q_value = (1 - learning_rate) * q_table[(q_table_row_index, 3)] + learning_rate * (reward + (discount_factor * q_table[(future_reward_row_index, 3)]))

    # Update Q-table
    q_table[(q_table_row_index, 3)] = q_value


def print_value(value_filename, policy_filename, q_filename, q_table):
    """Print the value, policy, and q values to .txt files

    :param value_filename: Path to value file output
    :param policy_filename: Path to policy file output
    :param q_filename: Path to q value file output
    :param q_table: Q-table array
    :return: None
    """
    best_indexes = []

    # Get best (max) rows for each state
    for i in range(0, len(q_table), 4):
        best_indexes += [i + np.argmax(q_table[i:i + 4, 3])]
    best_indexes = np.array(best_indexes)

    # Partition the q-table
    best_rows = q_table[best_indexes]
    values = best_rows[:, np.array([0, 1, 3])]
    policies = best_rows[:, np.array([0, 1, 2])]

    # Save everything to txt
    np.savetxt(value_filename, values, fmt='%s')
    np.savetxt(policy_filename, policies, fmt='%s')
    np.savetxt(q_filename, q_table, fmt='%s')


def main():
    # Retrieve command line args
    maze_input, value_file, q_value_file, policy_file, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon = sys.argv[1:]

    # Initialize the environment
    environment = env.Environment(maze_input)
    q_table = init_q_table(environment)

    # Train
    for episode in range(int(num_episodes)):
        print(f"Starting at: {environment.agent_location}")
        for episode_action in range(int(max_episode_length)):

            # Exploit
            if random.random() >= 1 - float(epsilon):
                action = get_optimal_action(q_table, environment.agent_location)

            # Explore
            else:
                action = random.randint(0, 3)

            # Retrieve <s, a, r, s'> tuple
            current_state = np.array(environment.agent_location)
            next_state, reward, is_terminal = environment.step(action)
            print(f"<{current_state}, {action}, {reward}, {next_state}>")

            # Update Q table with new tuple
            update_q_table(q_table, current_state, action, reward, next_state, float(learning_rate),
                           float(discount_factor))

            # Check if agent has reached the terminal state
            if is_terminal:
                print("Found the end.")
                environment.reset()
                break
            elif episode_action == int(max_episode_length) - 1:
                print("Did not find the end.")
                break

    # Print Q table to console and save tables to txt
    print_value(value_file, policy_file, q_value_file, q_table)
    print("Final Q-table:")
    print(q_table)
    return


if __name__ == "__main__":
    main()