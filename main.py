import gym
from scipy.misc import imresize
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np

from distributional_dqn import DistributionalDQN

IMAGE_SIZE = 80
ATOM_SIZE = 51
V_MIN = -10
V_MAX = 10
z = np.linspace(V_MIN, V_MAX, ATOM_SIZE)

env = gym.make('CartPole-v0')
env.reset()

agent = DistributionalDQN(1, env.action_space.n, ATOM_SIZE, IMAGE_SIZE, [5, 3, 5])


def select_action(z, probs):
    z_probs = np.multiply(probs, z)
    best_action = np.sum(z_probs, axis=1).argmax()
    return best_action


for _ in range(1000):
    rendered = env.render(mode='rgb_array')
    resized = imresize(rendered, [IMAGE_SIZE, IMAGE_SIZE])
    # img = Image.fromarray(resized, 'RGB').convert('LA')
    resized = np.reshape(resized, [1, 80, 80, 3]).astype(np.float32)
    resized = resized.transpose((0, 3, 2, 1))

    variable_input = Variable(torch.from_numpy(resized))

    probs = agent(variable_input)
    best_action = select_action(z, probs)
    observation, reward, done, _ = env.step(best_action)

    if done:
        env.reset()


