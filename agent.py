import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from replay_memory import ReplayMemory
from distributional_dqn import DistributionalDQN

EPSILON = 0.01
V_MIN = -1000
V_MAX = 10
ATOM_SIZE = 51
gamma = 0.90


class Agent:

    def __init__(self, max_memory, batch_size, action_size, atom_size, input_size, kernel_size):
        self.z = np.linspace(V_MIN, V_MAX, ATOM_SIZE)
        self.action_size = action_size
        self.epsilon = EPSILON
        self.batch_size = batch_size
        self.atom_size = atom_size
        self.memory = ReplayMemory(max_memory)
        self.brain = DistributionalDQN(action_size=action_size, atom_size=atom_size,
                                       input_size=input_size, kernel_size=kernel_size)
        self.target_brain = DistributionalDQN(action_size=action_size, atom_size=atom_size,
                                       input_size=input_size, kernel_size=kernel_size)
        self.target_brain.load_state_dict(self.brain.state_dict())
        self.optim = optim.Adam(self.brain.parameters(), lr=0.001)

    def step(self, state_input):
        probs = self.brain(state_input)
        best_action = self.select_best_action(probs)
        return best_action

    def select_best_action(self, probs):
        numpy_probs = self.variable_to_numpy(probs)
        # z_probs = np.multiply(numpy_probs, self.z)
        # best_action = np.sum(z_probs, axis=1).argmax()
        best_action = np.argmax(numpy_probs, axis=1)
        return best_action[0]

    def store_states(self, states, best_action, reward, done, next_states):
        td = self.calculate_td(states, best_action, reward, done, next_states)
        self.memory.add_memory(states, best_action, reward, done, next_states, td=td)

    def variable_to_numpy(self, probs):
        # probs is a list of softmax prob
        numpy_probs = probs.data.numpy()
        return numpy_probs

    #TODO find out why td does not get -100 reward
    def calculate_td(self, states, best_action, reward, done, next_states):
        probs = self.brain(states)
        numpy_probs = self.variable_to_numpy(probs)
        # states_prob = np.multiply(numpy_probs, self.z)
        # states_q_value = np.sum(states_prob, axis=1)[best_action]
        states_q_value = numpy_probs[0][best_action]

        next_probs = self.brain(next_states)
        numpy_next_probs = self.variable_to_numpy(next_probs)
        # next_states_prob = np.multiply(numpy_next_probs, self.z)
        # max_next_states_q_value = np.sum(next_states_prob, axis=1).max()
        max_next_states_q_value = np.max(numpy_next_probs, axis=1)[0]

        if done:
            td = reward - states_q_value
        else:
            td = (reward + gamma * max_next_states_q_value) - states_q_value

        return abs(td)

    def learn(self):
        # make sure that there is at least an amount of batch_size before training it
        if self.memory.count < self.batch_size:
            return

        tree_indexes, tds, batches = self.memory.get_memory(self.batch_size)
        total_loss = None
        for index, batch in enumerate(batches):

            # fixme fix this None type
            if batch is None:
                continue

            state_input = batch[0]
            best_action = batch[1]
            reward = batch[2]
            done = batch[3]
            next_state_input = batch[4]

            current_q = self.brain(state_input)
            # next_best_action = self.select_best_action(next_q)
            max_current_q = torch.max(current_q)

            best_next_action = self.select_best_action(self.brain(next_state_input))
            next_q = self.target_brain(next_state_input)
            # z_prob = self.variable_to_numpy(z_prob)

            target = reward + (1 - done) * gamma * next_q.data[0][best_next_action]
            target = Variable(torch.FloatTensor([target]))

            #TODO finish single dqn with per

            # target_z_prob = np.zeros([self.action_size, ATOM_SIZE], dtype=np.float32)
            # if done:
            #     Tz = min(V_MAX, max(V_MIN, reward))
            #     b = (Tz - V_MIN) / (self.z[1] - self.z[0])
            #     m_l = math.floor(b)
            #     m_u = math.ceil(b)
            #     target_z_prob[best_action][m_l] += (m_u - b)
            #     target_z_prob[best_action][m_u] += (b - m_l)
            # else:
            #     for z_index in range(len(z_prob)):
            #         Tz = min(V_MAX, max(V_MIN, reward + gamma * self.z[z_index]))
            #         b = (Tz - V_MIN) / (self.z[1] - self.z[0])
            #         m_l = math.floor(b)
            #         m_u = math.ceil(b)
            #
            #         target_z_prob[best_action][m_l] += z_prob[next_best_action][z_index] * (m_u - b)
            #         target_z_prob[best_action][m_u] += z_prob[next_best_action][z_index] * (b - m_l)
            # target_z_prob = Variable(torch.from_numpy(target_z_prob))

            # backward propagate
            output_prob = self.brain(batch[0])
            # loss = torch.sum(target_z_prob * torch.log(output_prob))

            loss = F.mse_loss(max_current_q, target)
            total_loss = -loss if total_loss is None else total_loss - loss

            # update td
            td = self.calculate_td(state_input, best_action, reward, done, next_state_input)
            tds[index] = td

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        # load brain to target brain
        self.target_brain.load_state_dict(self.brain.state_dict())

        self.memory.update_memory(tree_indexes, tds)
