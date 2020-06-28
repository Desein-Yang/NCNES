#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2020/02/01 18:01:12
@Describtion:   all functions about train and  test model in rl env    

"""

# here put the import lib

import time
import numpy as np
import torch
import os 
import copy

from gym import wrappers, logger
from src.model import build_model
# from test import build_model
from src.preprocess import ProcessUnit
from src.optimizer import check_bound



def get_reward_atari(
    model,
    mean,
    sigma,
    env,
    seed,
    ARGS,
    reference,
    no_op_frames=None,
    test=False,
    render=False):
    """
    Evaluates one model's offspring with Guassian nosie or evaluate itself.   
    Args:   
        model(nn.Module):   An initialized model to add noise and evaluate. If test is True, model is to be tested.
        mean(dict optional):Gauss distribution mean of a model.
        sigma(dict optional):Gauss distribution sigma of a model. 
        env(gym.Env):       Game environment for evaluation  
        ARGS:               All global setting such as timestep limit, fitness evaluated times, network settings   
        reference:          Visual batch normalizetion reference   
        no_op_frames(int or None,optional):  The number of n where n means don't take any action in first n frames.    if None, n is random number of (6,12,18,24,30).Default:None   
        test(bool,optional):If True, it is for test (fitness time = 1 and timesteps limit is 100000)  Default:False  
        render(bool,optional):If True, it is for save game record(don't use)   Default:False
    Returns:   
        ep_r(float32):      Mean reward of n repeated evaluations, n is ARGS.eva_times   
        frame_count(int):   Cumulative consumed frames of ft repeated evaluations
        noise(tensor):      List of sampled noise tensors in a population  
        model(nn.Module):   List of models in a population    
        no_op(int):         List of no-action frames number in each evaluation  
        ep_r_list(list):    List of rewards in each evaluation(detail of ep_r)
    """
    # log
    # 1. used for fixed noops   
    # 2. use add_noise_model function   
    # 3. return sigma and mean    
    # 4. record video
    # 5. don't use noise table

    # load mean model
    if (test is True) or (mean is None) or (sigma is None): 
        # when test model we do not need add noise
        eva_times = 1
        timestep_limit_episode = 100000
    else:
        # when train model we add Gaussian noise      
        with torch.no_grad():
            torch.manual_seed(seed)
            for name,params in model.named_parameters():
                A = torch.randn_like(params,dtype=torch.float) * sigma[name] + mean[name]
                params.data = torch.clamp(A,ARGS.L,ARGS.H)
                
        eva_times = ARGS.eva_times
        timestep_limit_episode = ARGS.timestep_limit_episode
                
    # first send in reference
    model.switch_to_bn()
    output = model(reference)
    # second switch to vbn
    model.switch_to_vbn()
    
    env.frameskip = 1
    ep_r_list = []
    frame_count = 0
    start_time = time.time()

    # init preprocess
    ProcessU = ProcessUnit()

    # repeated evaluation
    for times in range(eva_times):
        # use 6,12,18,24,30 as noop frames to decrease randomness by 6 times compared to 30-no-ops
        if no_op_frames == None:
            no_op = np.random.randint(1, 6) * 6
        else:
            no_op = no_op_frames * 6

        observation = env.reset()
        ProcessU.step(observation)
        break_is_true = False
        ep_r = 0.0

        for i in range(no_op):
            # 0 is Null Action but have not found any article about the meaning of every actions
            observation, reward, done, _ = env.step(0)
            ProcessU.step(observation)
            frame_count += 1

        for _ in range(timestep_limit_episode):
            action = model(ProcessU.to_torch_tensor())[0].argmax().item()
            for i in range(ARGS.FRAME_SKIP):
                observation, reward, done, _ = env.step(action)
                ProcessU.step(observation)
                frame_count += 1
                ep_r += reward
                if render:
                    env.render('rgb_array')
                if done:
                    break_is_true = True
            if break_is_true:
                break
        ep_r_list.append(ep_r)
    
    ep_r = sum(ep_r_list) / eva_times
    ep_time = int(time.time() - start_time)
    if (test is True) or (mean is None) or (sigma is None):
        return ep_r, frame_count, no_op, ep_r_list
    else:
        return ep_r, frame_count, model, no_op, ep_r_list, ep_time

def train(mean_list, sigma_list, pool, env, ARGS, refer_batch, seed):
    """Evaluates all offsprings of all populations in parallel by offspring seperately.   
    Args:   
        mean_list(list):    Gauss distribution mean (params state dict) of all population (lam x 1)   
        sigma_list(list):    Gauss distribution sigma (params state dict)of all population (lam x 1)   
        pool:                Thread pool for multiprocess   
        env:                 Game environment for evaluation   
        ARGS:                All global setting such as timestep limit, fitness evaluated times, network settings   
        refer_batch:         Visual batch normalizetion reference batch of all population   
    Returns:   
        rewards_list(list):     Reward means of all offsprings (lam x population_size)   
        frame_count(int):       Cumulative consumed frames of n repeated evaluations
        noises_list(list):      Sampled noise index of all offsprings (lam x population_size)    
        models_list(list):      All saved model (lam x population_size)  
        noops_list(list):       List of no op frame (lam x population_size)  
        detail_rewards_list(list):List of n repeated evaluation rewards (details of rewards_list) of all offsprings (lam x population_size x fitness_evaluate_times)   
    """
    jobs_list = []
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        jobs = None
        model = build_model(ARGS)
        #seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
        # create multiprocessing jobs
        jobs = [
                pool.apply_async(
                    get_reward_atari,
                    (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,)
                )
                for k_id in range(ARGS.population_size)
            ]        
        jobs_list.append(jobs)

    rewards_list = []
    frame_list = []
    models_list = []
    noops_list = []
    detail_rewards_list = []
    time_list = []
    # get reward(evaluate)
    for idx, jobs in enumerate(jobs_list):
        rewards ,frames, models, noops, detail_rewards, times= [],[],[],[],[],[]
        for j in jobs:
            rewards.append(j.get()[0])
            frames.append(j.get()[1])
            models.append(j.get()[2])
            noops.append(j.get()[3])
            detail_rewards.append(j.get()[4])
            times.append(j.get()[5])
        rewards_list.append(rewards)
        frame_list.append(frames)
        models_list.append(models)
        noops_list.append(noops)
        detail_rewards_list.append(detail_rewards)
        time_list.append(times)
        
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noops_list, detail_rewards_list, time_list

def train_serial(mean_list, sigma_list, env, ARGS, refer_batch, seed):
    """Evaluates all models one by one. """
    rewards_list, frame_list, models_list, noop_list, detail_rewards_list, times_list= [],[],[],[],[],[]
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        rewards ,frames, models, noops, detail_rewards, times= [],[],[],[],[],[]
        for k_id in range(ARGS.population_size):
            basemodel = build_model(ARGS)
            ep_r, frame, model, noop, ep_r_list = get_reward_atari(basemodel,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False)
            rewards.append(ep_r)
            frames.append(frame)
            models.append(model)
            noops.append(noop)
            detail_rewards.append(ep_r_list)
        rewards_list.append(rewards)
        frame_list.append(frames)
        models_list.append(models)
        noop_list.append(noops)
        detail_rewards_list.append(detail_rewards)
        times_list.append(times)
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noop_list, detail_rewards_list, times        
                  
def train_parallel(mean_list, sigma_list, pool, env, ARGS, refer_batch, seed):
    """Evaluates all offsprings of all populations in parallel by population seperately.""" 
    rewards_list,frame_list,models_list,noops_list,detail_rewards_list,times_list= [],[],[],[],[],[]
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        jobs = []
        model = build_model(ARGS)
        
        #seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
        # create multiprocessing jobs of population
        for k_id in range(ARGS.population_size):
            jobs.append(pool.apply_async(
                        get_reward_atari,
                        (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,)
                    )) 
        
        rewards ,frames, models, noops, detail_rewards,times= [],[],[],[],[],[]
        for j in jobs:
            rewards.append(j.get()[0])
            frames.append(j.get()[1])
            models.append(j.get()[2])
            noops.append(j.get()[3])
            detail_rewards.append(j.get()[4])
            times.append(j.get()[5])
        rewards_list.append(rewards)
        frame_list.append(frames)
        models_list.append(models)
        noops_list.append(noops)
        detail_rewards_list.append(detail_rewards)   
        times_list.append(times)
             
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noops_list, detail_rewards_list,times_list           

def train_individual(mean_list, sigma_list, pool, env, ARGS, refer_batch, seed):
    jobs = []
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        model = build_model(ARGS)
        #seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
        # create multiprocessing jobs
        for k_id in range(ARGS.population_size):
            jobs.append(pool.apply_async(
                        get_reward_atari,
                        (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,)
                    ))   

    rewards_list, frame_list, models_list, noops_list,detail_rewards_list, times_list = [],[],[],[],[],[]
    rewards ,frames, models, noops, detail_rewards, times= [],[],[],[],[],[]
        
    # get reward(evaluate)
    for idx,j in enumerate(jobs):
        rewards.append(j.get()[0])
        frames.append(j.get()[1])
        models.append(j.get()[2])
        noops.append(j.get()[3])
        detail_rewards.append(j.get()[4])
        times.append(j.get()[4])
    for i in range(ARGS.lam):
        mu = ARGS.population_size
        rewards_list.append(rewards[i * mu:(i+1) * mu])
        frame_list.append(frames[i * mu:(i+1) * mu])
        models_list.append(models[i * mu:(i+1) * mu])
        noops_list.append(noops[i * mu:(i+1) * mu])
        detail_rewards_list.append(detail_rewards[i * mu:(i+1) * mu])
        times_list.append(times[i * mu:(i+1) * mu])
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noops_list, detail_rewards_list, times_list

def train_individual_cpu(mean_list, sigma_list, pool, env, ARGS, refer_batch, seed):
    jobs = []
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        jobs = None
        model = build_model(ARGS)
        #seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
        # create multiprocessing jobs
        for k_id in range(ARGS.population_size):
            # jobs.append(pool.apply_async(
            #             get_reward_atari,
            #             (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,)
            #         ))   
            import affinity
            p = mp.Process(target = get_reward_atari,args = (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,))
            p.start()
            cpurank = affinity.get_process_affinity_mask(p.pid)  
            affinity.set_process_affinity_mask(p.pid,cpurank)
            p.join()

    rewards_list, frame_list, models_list, noops_list,detail_rewards_list = [],[],[],[],[]
    rewards ,frames, models, noops, detail_rewards= [],[],[],[],[]
        
    # get reward(evaluate)
    for idx,j in enumerate(jobs):
        rewards.append(j.get()[0])
        frames.append(j.get()[1])
        models.append(j.get()[2])
        noops.append(j.get()[3])
        detail_rewards.append(j.get()[4])
    for i in range(ARGS.lam):
        mu = ARGS.population_size
        rewards_list.append(rewards[i * mu:(i+1) * mu])
        frame_list.append(frames[i * mu:(i+1) * mu])
        models_list.append(models[i * mu:(i+1) * mu])
        noops_list.append(noops[i * mu:(i+1) * mu])
        detail_rewards_list.append(detail_rewards[i * mu:(i+1) * mu])
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noops_list, detail_rewards_list
	                  
def train_parallel_cpu(mean_list, sigma_list, pool, env, ARGS, refer_batch, seed):
    """Evaluates all offsprings of all populations in parallel by population seperately.""" 
    rewards_list,frame_list,models_list,noops_list,detail_rewards_list= [],[],[],[],[]
    for idx, mean in enumerate(mean_list):
        sigma = sigma_list[idx]
        jobs = []
        model = build_model(ARGS)
        
        #seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
        # create multiprocessing jobs of population
        for k_id in range(ARGS.population_size):
            #jobs.append(pool.apply_async(
            #            get_reward_atari,
            #            (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,)
            #        )) 
            p = mp.Process(target = get_reward_atari,args = (model,mean,sigma,env,seed[k_id],ARGS,refer_batch,None,False,False,))
            p.start()
            cpurank[k_id] = affinity.get_process_affinity_mask(p.pid)  
            affinity.set_process_affinity_mask(p.pid,cpurank)
            p.join()
            
        rewards ,frames, models, noops, detail_rewards= [],[],[],[],[]
        for j in jobs:
            rewards.append(j.get()[0])
            frames.append(j.get()[1])
            models.append(j.get()[2])
            noops.append(j.get()[3])
            detail_rewards.append(j.get()[4])
        rewards_list.append(rewards)
        frame_list.append(frames)
        models_list.append(models)
        noops_list.append(noops)
        detail_rewards_list.append(detail_rewards)   
             
    frame_count = np.sum(np.array(frame_list))
    return rewards_list, frame_count, models_list, noops_list, detail_rewards_list  

def test(model, pool, env, ARGS, reference, noop=None, test_times=30, render=False):
    """Evaluate all offsprings in parallel.   
    Args:   
        model(nn.Module):           Model to be tested   
        pool:                       Thread pool for multiprocess   
        env:                        Reinforcement game environment for evaluation   
        ARGS:                       All global setting such as timestep limit, fitness evaluated times, network settings   
        reference:                  Visual batch normalizetion reference   
        noop(int,optional):         No action frames, if None, noop is random number of (6,12,18,24,30) Default:None  
        test_times(int,optional):   Test times, if None, test times is 30. Default 30.
        render(bool, optional):     If true, record game.Default:False.  
    Returns:   
        rewards_list(float):  Mean reward of 30 repeated evaluation   
        times_list(list):    Consumed frames in 1 evaluation   
        noop_list(list):      List of noop frames  
        detail_rewards_list(list):   List of 30 repeated evaluation rewards (details of rewards_mean)    
    """
    # distribute training in parallel
    seed = [np.random.randint(1,1000000) for i in range(test_times)]
    jobs = [
        pool.apply_async(
            get_reward_atari,
            (model, None, None, env,seed[i], ARGS, reference, noop, True, render))
        for i in range(test_times)
    ]
    rewards_list = []
    times_list = []
    noop_list = []
    detail_rewards_list = []
    for idx, j in enumerate(jobs):
        rewards_list.append(j.get()[0])
        times_list.append(j.get()[1])
        noop_list.append(j.get()[2])
        detail_rewards_list.append(j.get()[3])
    return rewards_list,times_list, noop_list, detail_rewards_list





