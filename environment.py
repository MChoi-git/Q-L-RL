import numpy as np
import value_iteration as vi
import sys


class Environment:
    """Environment object which facilitates agent interactions during Q-learning.

    """
    def __init__(self, maze_filename):
        self.maze_filename = maze_filename
        self.maze = vi.init_maze_values(maze_filename)
        self.agent_location = np.concatenate(np.where(self.maze == -2))
        self.maze = np.where(self.maze == -2, -1, self.maze)

    def is_terminal(self, next_state):
        """Checks if agents next state s' is the terminal (goal) state.

        :param next_state: Tuple containing coordinates of the next state
        :return: Boolean
        """
        if self.maze[next_state] == 0:
            return True
        else:
            return False

    def process_movement(self, direction):
        """Determines parameters of the agent's next movement to state s', using action a.

        :param direction: The action the agent will take, coded as: [left, up, right, down]
        :return: The corresponding reward value for moving to the next state, boolean denoting if the next state is terminal, and
                 the coordinates of the next state
        """
        coding = np.array([[1, -1],     # Might be a better, more readable way to do this
                           [0, -1],
                           [1, 1],
                           [0, 1]])
        code = tuple(coding[direction])
        # Check that agent can move left
        if self.agent_location[code[0]] + code[1] >= 0 and ~np.isnan(self.agent_location[code[0]] + code[1]):
            self.agent_location[code[0]] += code[1]
            reward = self.maze[tuple(self.agent_location)]
            is_terminal = self.is_terminal(tuple(self.agent_location))
            next_state = self.agent_location
            return reward, is_terminal, next_state
        else:
            return -1.0, False, self.agent_location

    def step(self, action):  # Actions are: [left, up, right, down]
        """Take a step in the direction of action

        :param action: The action the agent will take, coded as: [left, up, right, down]
        """
        reward, is_terminal, next_state = self.process_movement(action)
        print(f"Next state properties: <{next_state}, {reward}, {int(is_terminal)}>")


def init_action_seq_file(actions_file):
    """Initializes the action sequence file into an iterable array.

    :param actions_file: File containing a sequence of actions from 0-4, separated by a space
    :return: An array of the sequence of actions
    """
    action_seq = np.genfromtxt(actions_file, delimiter=" ", dtype=int)
    return action_seq


# Retrieve command-line args
maze_input, output_file, action_seq_file = sys.argv[1:]
# Initialize the environment
environment = Environment(maze_input)
# Initialize the action sequence array
actions = init_action_seq_file(action_seq_file)
# Agent moves in the direction of each action
for i in actions:
    environment.step(i)

