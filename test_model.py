
import os
import click
import gym
import torch
import time
import pickle
import logging
import math
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import sys
from train import train, test
from model import build_model
from main import ARGS
import pickle
from vbn import explore_for_vbn
torch.set_num_threads(1)

Small_value = -1000000
def load_pickle(filename):
    df=open(filename,'rb')
    data3=pickle.load(df)
    df.close()
    return data3

def load_train(ARGS,save_path,gamename,logfile):

    env = gym.make(ARGS.gamename)
    ARGS.check_env(env)

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_best = build_model(ARGS)
    #print("resnet50 have {} paramerters in total".format(sum(x.numel() for x in model_best.parameters())))
    best_test_score = Small_value
    best_kid_mean = Small_value

    # init ARGS's parameter
    ARGS.set_logger(logger)
    ARGS.init(model_best.get_size(), params)
    ARGS.output()

    model_list = [build_model(ARGS) for i in range(ARGS.lam)]
    #model_list = [load('Seaquest-v0RUN-phi-0.001-seed-534484best_model.pt') for i in range(ARGS.lam)]
    sigma_list = [build_sigma(model_best, ARGS) for i in range(ARGS.lam)]
    model_optimizer_list = [SGD(model_best.named_parameters(), ARGS.lr_mean) for i in range(ARGS.lam)]
    sigma_optimizer_list = [SGD(model_best.named_parameters(), ARGS.lr_sigma, sigma=True) for i in range(ARGS.lam)]
    for model in model_list:
        model.set_parameter_no_grad()

    pool = mp.Pool(processes=ARGS.ncpu)

    refer_batch_torch = None
    if ARGS.env_type == "atari":
        # get reference batch
        logger.info("start testing reference batch statistic")
        reference_batch = explore_for_vbn(env, 0.01, ARGS)
        refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
        for i in range(ARGS.refer_batch_size):
            refer_batch_torch[i] = reference_batch[i]


    timestep_count = 0
    test_rewards_list = []
    break_training = False
    all_zero_count = 0
    for g in range(1):
        # fitness evaluation times
        rewards_list = [ [0] * ARGS.population_size ] * len(model_list)
        v = []
        seed_list = [v for i in range(len(model_list))]
        for i in range(ARGS.ft):
            one_rewards_list, one_seed_list, frame_count = train_simulate(model_list, sigma_list, pool, env, ARGS, refer_batch_torch)
            timestep_count += frame_count
            #logger.info("train_simulate:%s" % str(i))
            rewards_list += np.array(one_rewards_list)
            #logger.info("rewardkist%s"%str(i))
            for j,seed in enumerate(one_seed_list):
                seed_list[j].append(seed)
            #logger.info("seed%s"% str(i))
        rewards_list = rewards_list / ARGS.ft
    rewards_mean_list=[]
    for i in range(len(rewards_list)):
        rewards_mean_ = np.mean(np.array(rewards_list[i]))
        rewards_mean_list.append(rewards_mean_)

    with open(logfile,'a') as f:
        f.write(str(rewards_mean_list))
           
        

    # ---------------SAVE---------
    pool.close()
    pool.join()

if __name__ == "__main__":

    # set up multiprocessing
    mp.set_sharing_strategy("file_system")
    # log and save path setting
    torch.set_num_threads(1)
    # torch.manual_seed(int(time.time()))
    model_path = "log/2020-3-4-4/Qbert-phi-0.001-lam-5-mu-1526.pt"
    logfile = model_path[0:-3]+'.txt'

    gamename = "Qbert"
    
    ARGS.gamename = gamename + "NoFrameskip-v4"
    env = gym.make(ARGS.gamename)
    env.seed(int(time.time()))
    ARGS.action_n = env.action_space.n
    
    
    model = build_model(ARGS)
    model.load_state_dict(torch.load(model_path))
    pool = mp.Pool(processes=5)

    refer_batch_torch = None
    if ARGS.env_type == "atari":
        # get reference batch
        reference_batch = explore_for_vbn(env, 0.01, ARGS)
        refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
        for i in range(ARGS.refer_batch_size):
            refer_batch_torch[i] = reference_batch[i]

    test_rewards,test_timestep,test_noop_list_,_= test(model,pool,env,ARGS,refer_batch_torch,test_times=200)
    test_rewards_mean = np.mean(np.array(test_rewards))

    with open(logfile,'a') as f:
        f.write(str(test_rewards)+'\n')
        f.write('test final model mean'+','+str(test_rewards_mean)+'\n') 
    # ---------------SAVE---------
    pool.close()
    pool.join()
