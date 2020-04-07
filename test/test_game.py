import gym

def info_env(game):
    gamename = game + 'NoFrameskip-v4'
    env = gym.make(gamename)
    ac_n = env.action_space.n
    ob_shpae = env.observation_space.shape
    mes = env._max_episode_seconds
    mes2 = env._max_episode_steps
    es = env._elapsed_steps
    es2 = env._episode_started_at
    #config = env.configure
    ran = env.reward_range
    return ac_n,ob_shpae,mes,mes2,es,es2,ran

# game = 'Freeway'
# gamename = game + 'NoFrameskip-v4'
# env = gym.make(gamename)
# print(info_env(env))

def getattr(x):
    for each in x.__dir__():
        attr_name=each
        attr_value=x.__getattribute__(each)
        print(attr_name,':',attr_value)



import argparse
import sys

import gym
from gym import wrappers, logger

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def rend(game):

    env = gym.make(game+'NoFrameskip-v4')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './video/'+game
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    reward = 0
    done = False

    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        env.render('rgb_array')  
        if done:
            break
           

    # Close the env and write monitor result info to disk
    env.close()

a = {}
# for game in ['Freeway','Enduro','BeamRider','Alien','Breakout','SpaceInvaders','Venture','Seaquest','Pong','MontezumaRevenge','Pitfall']:
for game in ['Freeway','Enduro','BeamRider']:
    # a[game] = info_env(game)
    rend(game)
