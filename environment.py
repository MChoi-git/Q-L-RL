import numpy as np
import value_iteration as vi
import sys


class Environment:
    def __init__(self, maze_filename):
        self.maze_filename = maze_filename
        self.maze = vi.init_maze_values(maze_filename)
        self.agent_location = np.concatenate(np.where(self.maze == -2))
        self.maze = np.where(self.maze == -2, -1, self.maze)

    def is_terminal(self, next_state):
        if self.maze[next_state] == 0:
            return True
        else:
            return False

    def process_movement(self, direction):
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
        reward, is_terminal, next_state = self.process_movement(action)
        print(f"Next state properties: <{next_state}, {reward}, {int(is_terminal)}>")
        return


def init_action_seq_file(actions_file):
    action_seq = np.genfromtxt(actions_file, delimiter=" ", dtype=int)
    return action_seq


maze_input, output_file, action_seq_file = sys.argv[1:]
environment = Environment(maze_input)
print(environment.maze)
print(environment.agent_location)
actions = init_action_seq_file(action_seq_file)
for i in actions:
    environment.step(i)

