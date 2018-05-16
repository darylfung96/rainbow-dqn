import gym
from scipy.misc import imresize
from PIL import Image

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    rendered = env.render(mode='rgb_array')
    resized = imresize(rendered, [80, 80, 30])
    img = Image.fromarray(resized, 'RGB').convert('LA')
    env.step(env.action_space.sample())