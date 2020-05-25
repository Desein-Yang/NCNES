import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
import copy
import math
# from nn_builder.pytorch.NN import NN
from src.model import CNNModel,MLPModel,ESModel
from src.preprocess import ProcessUnit
from src.demo import Buffer
from torch.optim import SGD
from torch import nn
from torch.distributions import Categorical,Normal
import torch.nn.functional as F
from torchvision import transforms
from collections import deque

class Worker(object):
    """worker for rollout in enviroment"""
    def __init__(self,env,ARGS):
        self.env = env   
        # self.actor_model = MLPModel(ARGS.FRAME_SKIP, ARGS.action_n, True)
        # self.critic_model = MLPModel(ARGS.FRAME_SKIP, 1, False)
        self.actor_model = ESModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic_model = ESModel(ARGS.FRAME_SKIP,1,False)
        self.reset_point = ARGS.T
        self.timestep_limit = ARGS.timestep_limit
        self.gamma = ARGS.gamma
        self.eps = ARGS.epsilon
        self.ProcessU = ProcessUnit()

    def update(self,actor_model,critic_model,reset_point):
        self.actor_model.load_state_dict(actor_model)
        self.critic_model.load_state_dict(critic_model)
        self.reset_point = reset_point

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def noop_reset(self,noop_max=30):
        for i in range(noop_max):
        # 0 is Null Action but have not found any article about the meaning of every actions
            ob, reward, done, info = env.step(0)
            self.ProcessU.step(observation)
            step += 1
        info['step'] = step
        return ob, reward, done, info

    def rollout(self,demo,ARGS):
        """Rollout worker for evaluation and return batch D and success times W"""
        # TODO: reset to state function
        # get action()
        # replay demo()
        L = ARGS.batch_rollout_size
        K = ARGS.rnn_memory_size
        d = ARGS.nums_start_point
        W = 0 # success rate W 
        D = Buffer()

        reset_point = self.reset_point
        start_point = int(np.random.uniform(reset_point - d,reset_point)) 
        # start = tao * reset = tao
        self.env.reset(start_point)
        # ob, reward, done, info = self.noop_reset()
        # Replay env has used noop
        i = start_point - K
        frame_count = 0

        # pdb.set_trace()
        for step in range(L):
            # ARGS.logger.log_for_debug('start point%d' % start_point)
            # ARGS.logger.log_for_debug('start point%d' % reset_point)

            if i > start_point:
                # sample actionn
                action = self.actor_model(self.ProcessU.to_torch_tensor())[0].argmax().item()
                reward = 0
                for _ in range(ARGS.FRAME_SKIP):
                    ob, r, done, info = self.env.step(action)
                    self.ProcessU.step(ob)
                    reward += r
                frame_count += ARGS.FRAME_SKIP
                is_train = True
            else:
                action, ob, reward, done, info = demo.replay(i)
                reward = 0
                for _ in range(ARGS.FRAME_SKIP):
                    ob, r, done, info = self.env.step(action)
                    self.ProcessU.step(ob)
                    reward += r
                is_train = False
            info['step'] = step
            # pdb.set_trace()

            ob = self.ProcessU.to_torch_tensor()
            D.add(action,ob,reward,done,info,is_train)
            i = i + 1

            if done:
                if sum(D.rewards[reset_point:]) > sum(demo.rewards[reset_point:]):
                    W = W + 1
                start_point = int(np.random.uniform(reset_point - d,reset_point))
                i = start_point - K
                self.env.reset(start_point)      
            # pdb.set_trace()

        return D,W,frame_count


class PPO_Optimizer(object):
    def __init__(self,ARGS):
        # ob_space = self.env.observation_space
        # ac_space = self.env.action_space

        # ARGS.action_n = self.env.action_space.n
        self.adv = []
        self.eps = ARGS.eps
        self.w_vf = ARGS.weight_vf
        self.w_ent = ARGS.weight_entropy
        self.gamma = ARGS.gamma
        self._max_grad_norm = ARGS.max_grad_norm

        self.actor_old = ESModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic_old = ESModel(ARGS.FRAME_SKIP,1,False)
        self.actor = ESModel(ARGS.FRAME_SKIP,ARGS.action_n,True)
        self.critic = ESModel(ARGS.FRAME_SKIP,1,False)
        self.actor_optim = SGD(self.actor.parameters(), lr=ARGS.lr)
        self.critic_optim = SGD(self.critic.parameters(), lr=ARGS.lr)
        self.ProcessU = ProcessUnit()
        self.logger = ARGS.logger

    # Finished
    def get_advantage_est(self,buffer):
        """get advantage value estimation as equation (10)"""
        adv = []      
        discount_r = self.get_discounted_reward(buffer)
        # target_v = self.get_target_v(buffer)
        state_v = self.get_state_v(buffer)
        
        # assert len(buffer.log_prob) == len(buffer.discount_r)
        # for state_value,d_reward in zip(buffer.state_value,buffer.discount_r):
        for s_v, t_v in zip(state_v, discount_r):
            adv.append(t_v - s_v)
        adv = torch.cat(adv)
        return adv

    # Finished
    def get_discounted_reward(self,buffer,norm=True):
        """return [R[0],R[1]...R[n]]"""
        discount_r = []
        import pdb; pdb.set_trace()
        # from torch.autograd import Variable
        # ob = Variable(torch.from_numpy(buffer.obs[-1]),requires_grad = False)
        # ob = ob.byte()
        # ob = torch.tensor(torch.from_numpy(buffer.obs[-1])).clone().detach().requires_grad_(False)

        ob = buffer.obs[-1]
        value = self.critic_old.forward(ob)
        for reward in buffer.rewards:
            discount_r.insert(0,value)
            value += value * self.gamma + reward
        discount_r = torch.tensor(discount_r)
        if norm:
            discount_r = (discount_r - discount_r.mean()) / (discount_r.std() + self.eps)
        buffer.discount_r = discount_r.clone()
        return discount_r

    def get_target_v(self,buffer):
        tmp = [0]
        for idx,ob in enumerate(buffer.obs):          
            if idx + 1 < buffer.size:
                tmp.append(self.critic_old.forward(buffer.obs[idx+1]))
        target_v = buffer.returns + self.gamma * tmp
        return target_v    

    def get_state_v(self,buffer):
        for ob in buffer.obs:          
            buffer.state_value.append(self.critic.forward(ob))
        return buffer.state_value

    def optimize(self,actor_model,critic_model,D):
        self.logger.log_for_debug('190')

        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.actor.load_state_dict(actor_model.state_dict())
        self.critic.load_state_dict(critic_model.state_dict()) 
        
        self.logger.log_for_debug('196')
        self.get_discounted_reward(D)
        adv = self.get_advantage_est(D)
        for idx in range(D.size):
            action = torch.tensor(D.actions[idx],dtype=torch.uint8)
            ob = D.obs[idx]
            self.logger.log_for_debug('206')
            dist = Categorical(self.actor.forward(ob))
            dist_old = Categorical(self.actor_old.forward(ob))
            self.logger.log_for_debug('206')

            ratio = torch.exp(dist.log_prob(action) - dist_old.log_prob(action))
            surr1 = ratio * adv
            surr2 = ratio.clamp(1. - self.eps, 1. + self.eps ) * adv
            self.logger.log_for_debug('209')

            clip_loss = - torch.min(surr1,surr2).mean()
            clip_losses.append(clip_loss.numpy())

            vf_loss = F.smooth_l1_loss(self.critic(ob),D.state_value)
            vf_losses.append(vf_loss.numpy())

            ent_loss = dist.entropy().mean()
            ent_losses.append(ent_loss.numpy())

            loss = clip_loss + self.w_vf * vf_loss + self.w_ent * ent_loss
            losses.append(loss.numpy())
            self.logger.log_for_debug('219')

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + 
                list(self.critic.parameters()),
            self._max_grad_norm)
            self.logger.log_for_debug('228')

            # update parameter of actor and critic
            self.actor_optim.step()
            self.critic_optim.step()

        lossdict = {
            'loss' : losses,
            'Cliploss' : clip_losses,
            'VF  loss' : vf_losses,
            'Ent loss' : ent_losses,
        }
        return lossdict, self.actor.state_dict(), self.critic.state_dict()
