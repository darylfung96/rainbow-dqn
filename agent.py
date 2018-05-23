import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

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
        self.action_size = action_size
        self.epsilon = EPSILON
        self.batch_size = batch_size
        self.memory = ReplayMemory(max_memory)
        self.brain = DistributionalDQN(action_size=action_size, atom_size=atom_size,
                                       input_size=input_size, kernel_size=kernel_size)
        self.target_brain = DistributionalDQN(action_size=action_size, atom_size=atom_size,
                                       input_size=input_size, kernel_size=kernel_size)
        self.criterion = nn.CrossEntropyLoss()

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
        self.memory.add_memory(states, best_action, reward, done, next_states, td=td)

    def learn(self):
        # make sure that there is at least an amount of batch_size before training it
        if self.memory.count < self.batch_size:
            return

        tree_indexes, batches = self.memory.get_memory(self.batch_size)
        total_loss = None

        for batch in batches:

            #fixme fix this None type
            if batch is None:
                continue

            state_input = batch[0]
            best_action = batch[1]
            next_state_input = batch[4]

            next_q = self.brain(next_state_input)
            next_best_action = self.select_best_action(next_q)

            z_prob = self.target_brain(state_input)

            target_z_prob = np.zeros([self.action_size, ATOM_SIZE])

            for z_index in range(len(z_prob)):
                Tz = min(V_MAX, max(V_MIN, batch[2] + gamma * self.z[z_index]))
                b = (Tz - V_MIN) / (self.z[1] - self.z[0])
                m_l = math.floor(b)
                m_u = math.ceil(b)

                target_z_prob[best_action][m_l] += z_prob[next_best_action][z_index] * (m_u - b)
                target_z_prob[best_action][m_u] += z_prob[next_best_action][z_index] * (b - m_l)
            target_z_prob = Variable(torch.from_numpy(target_z_prob))

            # backward propagate
            #TODO fix error target_z_prob not a longtensor
            output_prob = self.brain(batch[0])
            output_prob = Variable(torch.from_numpy(output_prob), requires_grad=True)
            loss = torch.sum(target_z_prob * torch.log(output_prob))
            total_loss = loss if total_loss is None else total_loss + loss

        total_loss.backward()


