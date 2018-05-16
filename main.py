import gym
from scipy.misc import imresize
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np

from distributional_dqn import DistributionalDQN

env = gym.make('CartPole-v0')
env.reset()

agent = DistributionalDQN(1, env.action_space.n, 80, [5, 3, 5])

for _ in range(1000):
    rendered = env.render(mode='rgb_array')
    resized = imresize(rendered, [80, 80, 30])
    resized = np.reshape(resized, [1, 80, 80, 3]).astype(np.float32)
    resized = resized.transpose((0, 3, 2, 1))

    variable_input = Variable(torch.from_numpy(resized))

    # img = Image.fromarray(resized, 'RGB').convert('LA')
    action = agent(variable_input)
    best_action = np.argmax(action)
    env.step(best_action)