import gym
from scipy.misc import imresize
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np

from agent import Agent

MAX_EPISODE = 10000
IMAGE_SIZE = 80
ATOM_SIZE = 51

ALPHA = 0.6
BETA = 0.4


env = gym.make('CartPole-v0')
variable_input = env.reset()


def preprocess_image(original_image):
    resized = imresize(original_image, [IMAGE_SIZE, IMAGE_SIZE])
    # img = Image.fromarray(resized, 'RGB').convert('LA')
    resized = np.reshape(resized, [1, 80, 80, 3]).astype(np.float32)
    resized = resized.transpose((0, 3, 2, 1))

    variable_input = Variable(torch.from_numpy(resized))
    return variable_input


agent = Agent(max_memory=200, batch_size=32, action_size=env.action_space.n, atom_size=ATOM_SIZE, input_size=IMAGE_SIZE,
              kernel_size=[5, 3, 5])
total_reward = 0
rendered = env.render(mode='rgb_array')
for current_episode in range(MAX_EPISODE):
    # variable_input = preprocess_image(rendered)
    variable_input = np.reshape(variable_input, [1, -1])
    variable_input = Variable(torch.from_numpy(variable_input).float())
    best_action = agent.step(variable_input)
    next_input, reward, done, _ = env.step(best_action)
    reward = reward if not done else -100
    total_reward += reward
    next_input = np.reshape(next_input, [1, -1])
    next_variable_input = Variable(torch.from_numpy(next_input).float())

    if done:
        print('total reward: ', total_reward)
        total_reward = 0
        next_input = env.reset()
        next_input = np.reshape(next_input, [1, -1])
        next_variable_input = Variable(torch.from_numpy(next_input).float())
        agent.learn()

    # next_rendered = env.render(mode='rgb_array')
    # next_variable_input = preprocess_image(next_rendered)
    env.render()
    agent.store_states(variable_input, best_action, reward, done, next_variable_input)
    variable_input = next_input


env.close()



