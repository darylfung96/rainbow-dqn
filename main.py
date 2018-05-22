import gym
from scipy.misc import imresize
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np

from distributional_dqn import DistributionalDQN
from agent import Agent

MAX_EPISODE = 1000
IMAGE_SIZE = 80
ATOM_SIZE = 51

ALPHA = 0.6
BETA = 0.4


env = gym.make('CartPole-v0')
env.reset()


def preprocess_image(original_image):
    resized = imresize(original_image, [IMAGE_SIZE, IMAGE_SIZE])
    # img = Image.fromarray(resized, 'RGB').convert('LA')
    resized = np.reshape(resized, [1, 80, 80, 3]).astype(np.float32)
    resized = resized.transpose((0, 3, 2, 1))

    variable_input = Variable(torch.from_numpy(resized))
    return variable_input


agent = Agent(max_memory=1000, batch_size=1, action_size=env.action_space.n, atom_size=ATOM_SIZE, input_size=IMAGE_SIZE,
              kernel_size=[5, 3, 5])

rendered = env.render(mode='rgb_array')
for _ in range(MAX_EPISODE):
    variable_input = preprocess_image(rendered)
    best_action = agent.step(variable_input)
    observation, reward, done, _ = env.step(best_action)

    if done:
        env.reset()

    next_rendered = env.render(mode='rgb_array')
    next_variable_input = preprocess_image(next_rendered)
    agent.store_states(variable_input, best_action, reward, done, next_variable_input)



