import numpy as np
from replay_memory import ReplayMemory
from distributional_dqn import DistributionalDQN

EPSILON = 0.01
V_MIN = -10
V_MAX = 10
ATOM_SIZE = 51
gamma = 0.90


class Agent:

    def __init__(self, max_memory, batch_size, action_size, atom_size, input_size, kernel_size):
        self.z = np.linspace(V_MIN, V_MAX, ATOM_SIZE)
        self.epsilon = EPSILON
        self.batch_size = batch_size
        self.memory = ReplayMemory(max_memory)
        self.brain = DistributionalDQN(action_size=action_size, atom_size=atom_size,
                                       input_size=input_size, kernel_size=kernel_size)

    def step(self, state_input):
        probs = self.brain(state_input)
        best_action = self.select_best_action(probs)
        return best_action

    def select_best_action(self, probs):
        z_probs = np.multiply(probs, self.z)
        best_action = np.sum(z_probs, axis=1).argmax()
        return best_action

    def store_states(self, states, best_action, reward, done, next_states):
        states_prob = np.multiply(self.brain(states), self.z)
        states_q_value = np.sum(states_prob, axis=1)[best_action]

        next_states_prob = np.multiply(self.brain(next_states), self.z)
        max_next_states_q_value = np.sum(next_states_prob, axis=1).max()
        td = ( reward + gamma * max_next_states_q_value ) - states_q_value
        self.memory.add_memory(states, reward, done, next_states, td=td)

    def learn(self):
        # make sure that there is at least an amount of batch_size before training it
        if self.memory.count < self.batch_size:
            return

        tree_indexes, batches = self.memory.get_memory(self.batch_size)
        pass

