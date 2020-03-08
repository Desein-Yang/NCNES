#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2020/02/01 17:58:21
@Describtion:   main function to run experiment
"""

# here put the import lib

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
import sys
import matplotlib.pyplot as plt

from optimizer import optimize_parallel
from train import train,test
from model import build_model
from util import mk_folder, save, load, setup_logging
from vbn import explore_for_vbn

# set up multiprocessing
mp.set_sharing_strategy("file_system")
# log and save path setting
torch.set_num_threads(1)


class ARGS(object):
    """
    Global shared setting.   
    """

    env_type = "atari"

    state_dim = 0
    action_dim = 0
    action_lim = 0

    # general setting from CES
    # fixed
    timestep_limit = int(1e8)
    timestep_limit_episode = 100000
    test_times = 30

    # input parameters
    namemark = ""
    ncpu = 0
    population_size = 0
    gamename = ""
    eva_times = 0
    sigma_init = 0.0
    lam = 0
    phi = 0.0
    lr_mean = 0.0
    lr_sigma = 0.0

    # fixed parameters
    l2coeff = 0.005
    generation = 5000
    H = 10
    L = -10
    FRAME_SKIP = 4
    action_n = 0
    refer_batch_size = 128

    phi_decay = True
    lr_decay = True
    logger = None
    folder_path = os.getcwd()
    checkpoint_name = ""
    logfile_name = ""
    Small_value = -1000000

    @classmethod
    def output(cls):
        """output basic information of one run"""
        logger = cls.logger
        logger.info("envtype:%s" % cls.env_type)
        logger.info("Gamename:%s" % cls.gamename)
        logger.info("lambda:%s" % cls.lam)
        logger.info("population size:%s" % cls.population_size)
        logger.info("phi:%s" % cls.phi)
        logger.info("lr_mean:%s" % cls.lr_mean)
        logger.info("lr_sigma:%s" % cls.lr_sigma)
        logger.info("sigma_init:%s" % cls.sigma_init)
        logger.info("timestep limit:%s" % cls.timestep_limit)
        logger.info("lr decay enable ?:%s" % cls.lr_decay)
        logger.info("phi decay enable ?:%s" % cls.phi_decay)
        logger.info("H: %s; L: %s" % (cls.L, cls.H))
        logger.info("EvaluateTimes %s" % cls.eva_times)

    @classmethod
    def set_params(cls, env, kwargs):
        """Set up hyperparameters in ARGS class"""
        # cls.eva_times = kwargs["eva_times"]
        cls.phi = kwargs["phi"]
        cls.lam = kwargs["lam"]
        cls.population_size = kwargs["mu"]
        cls.lr_mean = kwargs["lr_mean"]
        cls.sigma_init = kwargs["sigma_init"]
        cls.lr_sigma = kwargs["lr_sigma"]

        cls.action_n = env.action_space.n
        cls.checkpoint_name = cls.gamename.split('N')[0]+"-phi-" + str(cls.phi) + "-lam-" + str(cls.lam) + "-mu-" + str(cls.population_size)
        gamename = cls.gamename.split('N')[0]
        if gamename == "Alien" or 'Qbert' or 'SpaceInvaders':
            cls.eva_times = 10
        elif gamename == 'Breakout' or 'Seaquest' or 'Freeway':
            cls.eva_times = 1
        else:
            cls.eva_times = 3
        if gamename == 'SpaceInvaders':
            cls.FRAME_SKIP = 3
    
    @classmethod
    def set_recommend(cls,model_size):
        """set up recommmand set"""
        # cls.lr_mean = 1
        # cls.lr_sigma = 0.001
        pass
        # cls.lr_sigma = (3+math.log(model_size)) / 5 / math.sqrt(model_size)
        # cls.sigma_init = (cls.H - cls.L) / cls.lam

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger
    
    @classmethod
    def set_gamename(cls, gamename):
        if cls.env_type == "atari":
            cls.gamename = "%sNoFrameskip-v4" % gamename
    
    @classmethod
    def set_folder_path(cls,folder_path):
        cls.folder_path = folder_path
    
    @classmethod
    def set_logfile_name(cls,logfile_name):
        cls.logfile_name = cls.gamename.split('N')[0] + cls.namemark + "-phi-" + str(cls.phi) + "-lam-" + str(cls.lam) + "-mu-" + str(cls.population_size)+".txt"



def build_sigma(model: torch.nn.Module, ARGS):
    """Build a dict to store sigma of all params.  
    Args:  
        model(nn.Module):   Network module of offspring.
        ARGS:               Sigma init value.
    Returns:  
        sigma_dict(dict):   Dict of sigma of all params.
    Init:  
        ones_like tensor * sigma_init.
    """
    sigma_dict = {}
    for name, parameter in model.named_parameters():
        sigma_dict[name] = torch.ones_like(parameter,dtype = torch.float) * ARGS.sigma_init
    return sigma_dict

def build_mean(model: torch.nn.Module,ARGS):
    """Build a dict to store mean of all params.  
    Args:  
        model(nn.Module):   Network module of offspring.
        ARGS:               High limit and low limit
    Returns:  
        mean_dict(dict):    Dict of mean of all params.  
    Init:
        mean= L + (H-L) *rand
    """
    mean_dict = {}
    for name, parameter in model.named_parameters():
        mean_dict[name] = torch.ones_like(parameter,dtype=torch.float) * ARGS.L + (ARGS.H - ARGS.L) * torch.rand_like(parameter,dtype=torch.float)
        mean_dict[name] = torch.clamp(mean_dict[name],ARGS.L,ARGS.H)
    return mean_dict

def main(ARGS, logger, params):
    """Algorithms main procedures     
    Args:     
        params(dict):   Hyperparams (namemark,ncpu,envs_list,mu,lam,sigma_init,lr_mean,lr_sigma,eva_times,phi)
        {
            namemark:run1,
            ncpu:40,
            ...
        }
    """
    logger.info("Game name: %s", ARGS.gamename)
    logger.info("ncpu:%s", ARGS.ncpu)
    logger.info("namemark:%s", ARGS.namemark)

    # init rl environment 
    env = gym.make(ARGS.gamename)

    # init ARGS's parameter
    ARGS.set_logger(logger)
    ARGS.set_params(env,params)

    # set up random seed 
    # torch.manual_seed(time.time())
    # np.random.seed(123456)
    # seed = np.random.randint(0, 1000000)
    # env.seed(123456)

    # init best and timestep
    best_test_score = ARGS.Small_value
    best_train_score = ARGS.Small_value
    timestep_count = 0
    model_best = build_model(ARGS)
    model_best.set_parameter_no_grad()
    model_size = model_best.get_size()
    ARGS.set_recommend(model_size)
    ARGS.output()

    # init population
    mean_list = [build_mean(model_best,ARGS) for i in range(ARGS.lam)]
    sigma_list = [build_sigma(model_best, ARGS) for i in range(ARGS.lam)]

    # init pool
    pool = mp.Pool(processes=ARGS.ncpu)

    # vitural batch normalization
    refer_batch_torch = None
    if ARGS.env_type == "atari":
        # get reference batch
        logger.info("start testing reference batch statistic")
        reference_batch = explore_for_vbn(env, 0.01, ARGS)
        refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
        for i in range(ARGS.refer_batch_size):
            refer_batch_torch[i] = reference_batch[i]

    for g in range(ARGS.generation):
        if ARGS.lr_decay:
            percent = timestep_count / ARGS.timestep_limit
            ARGS.lr_mean = ARGS.lr_mean * (math.e - math.exp(percent)) / (math.e - 1)
            ARGS.lr_sigma = ARGS.lr_sigma * (math.e - math.exp(percent)) / (math.e - 1)


        if ARGS.phi_decay:
            ARGS.phi = ARGS.phi * (math.e - math.exp(percent)) / (math.e - 1)

        # sample and evaluate
        rewards_list, frame_count, models_list, noops_list, detail_rewards = train(
            mean_list,
            sigma_list,
            pool,env,
            ARGS,
            refer_batch_torch
        )
        timestep_count += frame_count
        logger.info("=============================================")
        logger.info("Gen            :%2d " % g)
        logger.info("Framecount     :%9d " % (frame_count))
        logger.info("AllFramecount  :%s/%s" % (timestep_count,ARGS.timestep_limit))
        logger.info("Rewardlist     :%s " % str(rewards_list))
        logger.info("Rewardlist mean:%s " % str(np.mean(np.array(rewards_list))))
        logger.info("Rewardlist var:%s " % str(np.var(np.array(rewards_list))))
        # logger.info("Noopslist      :%s " % str(noops_list))
        # logger.info("DetailReward   :%s " % str(detail_rewards))
        
        # save best one model
        index = np.array(rewards_list).argmax()
        best_model_i, best_model_j = (
            index // ARGS.population_size,
            index % ARGS.population_size,
        )

        if rewards_list[best_model_i][best_model_j] > best_train_score:
            best_train_score = rewards_list[best_model_i][best_model_j]
            logger.info("BestTrainScore:%.1f " % (best_train_score))
            


            # logger.info("Noopslist(New)  :%s  " % str(test_noop_list))
            
            # Update best model only when its test fitness higher than historic one
            test_rewards,test_timestep,test_noop_list,_= test(models_list[best_model_i][best_model_j],pool,env,ARGS,refer_batch_torch)
            best_test_new = np.mean(np.array(test_rewards))            
            if best_test_new > best_test_score:
                # save best model
                best_test_score = best_test_new
                savepath = save(models_list[best_model_i][best_model_j],ARGS.checkpoint_name,ARGS.folder_path,g)
                model_best.load_state_dict(torch.load(savepath))
                logger.info("BestTest(New)   :%.1f" % (best_test_score))
                logger.info("Rewardlist(New) :%s  " % str(test_rewards))
                logger.info("Update best model")
            
        if g % 5 == 0 :
            # test current best model
            test_rewards,test_timestep,test_noop_list,_= test(model_best,pool,env,ARGS,refer_batch_torch)
            test_rewards_mean = np.mean(np.array(test_rewards))
            logger.info("BestGen(test)    :%.1f" % (g))
            logger.info("TestModel(test)  :%.1f" % (test_rewards_mean))
            logger.info("Rewardlist(test) :%s  " % str(test_rewards))



        # calculate gradient and update distribution in parallel
        mean_list, sigma_list = optimize_parallel(g,mean_list,sigma_list,models_list,rewards_list,pool,ARGS)
     
        
        # log for train curve
        path = os.path.join(ARGS.folder_path,'train_curve.txt')
        with open(path, "a") as f:
            f.write(str(g) + "," + str(timestep_count)+ "," + str(best_train_score) + "," + str(best_test_new)+'\n')

        # check timestep
        if timestep_count > ARGS.timestep_limit:
            logger.info("Satisfied timestep limit")
            break

    # test final best model
    test_rewards,test_timestep,test_noop_list_,_= test(model_best,pool,env,ARGS,refer_batch_torch,test_times=30)
    test_rewards_mean = np.mean(np.array(test_rewards))
    logger.info("BestTest(Final) :%.1f" % (test_rewards_mean))
    logger.info("Rewardlist(Final):%s  " % str(test_rewards))
    
    pool.close()
    pool.join()
    savepath = save(model_best, ARGS.checkpoint_name, ARGS.folder_path,g)


def tune_params(
    namemark,
    ncpu,
    env_list,
    mu=15,
    lam=4,
    sigma_init=1,
    lr_mean=0.2,
    lr_sigma=0.01,
    phi=0.0001,
):
    """Set parameters for tuning.   
    Set up logger and folder path.    
    Run main().  
    Args:  
        namemark(str): Name of log file.  
        ncpu(int):     Number of availble CPU.  
        env_list(list):Environment name of games.  
        mu(int):       Population size.  Default:15.  
        lam(int):      Numbers of population. Default:4.  
        sigma_init(float): Init value of sigma.Default:1.  
        lr_mean(float):    Learning rate of mean. Default:0.2.
        lr_sigma(float):   Learning rate of sigma. Default:0.01.  
        eva_times(int):    Evaluate times. Default:3.  
        phi(float):        Negative correlation.
    """
    # set input parameters
    ARGS.env_type = "atari"
    ARGS.namemark = namemark
    ARGS.ncpu = ncpu

    kwargs_list = []
    kwargs_list.append(
        {
            "phi": phi,
            "mu": mu,
            "lam": lam,
            "sigma_init": sigma_init,
            "lr_mean": lr_mean,
            "lr_sigma": lr_sigma,
        }
    )
    
    # set folder path
    folder_path = mk_folder(os.path.join(os.getcwd(), "log"))
    print("start!")

    idx = 0
    print("-----------------------------------------")
    print("the number of hyperparameters combination:", len(kwargs_list))

    # set logger handler and run main
    logger = logging.getLogger(__name__)
    for game in env_list:
        for params in kwargs_list:
            ARGS.set_gamename(game)
            ARGS.set_folder_path(folder_path)
            logfile = (namemark + "-" + game + "-phi-" + str(params["phi"])+'.txt')
            logger = setup_logging(logger, folder_path, logfile)
            main(ARGS, logger, params)
            print("finish idx %s : %s for game:%s" % (idx, str(params), game))
            idx += 1
            draw_train_curve(game,ARGS)

def draw_train_curve(game,ARGS):
    path = ARGS.folder_path + "train_curve.txt"
    
    with open(path,'r') as f:
        lines = f.readlines()
        gen = []
        for i,line in enumerate(lines):
            line_split = line.split(',')
            gen.append(int(line_split[0]))
            timestep.append(int(line_split[1]))
            best_train_score.append(float(line_split[2]))
            best_test_score.append(float(line_split[3]))


    fig, ax = plt.subplots()
    ax.plot(timestep,best_train_score,label='model_best')
    ax.plot(timestep,best_test_score,label='test_best')
    
    title = game
    plt.title(title)
    plt.xlabel('timestep')
    plt.ylabel('score')
    plt.legend()
    plt.savefig(title+'-'+str(ARGS.phi)+'-'+'.png')
    plt.show()

if __name__ == "__main__":
    # gamelist
    # envs_list = ["Qbert", "Enduro", "Breakout", "Venture", "Freeway",'Alien','Seaquest','SpaceInvaders']
    envs_list = ["Freeway"]

    # parameters:
    namemark = "tuning"
    ncpu = 85
    mu = 15  # population_size
    lam = 5  # numbers of population
    sigma_init = 2
    lr_mean = 0.2
    lr_sigma = 0.1
    phi = 0.0001 

    # seaquest lr_mean = 1 lr_sigma = 1 phi = 0.001 eva = 3 lam = 4 mu = 4
    # alien lr_mean = 1
    # qbert lr_sigma = 0.1 lr_mean = 0.2 phi = 0.001 5300
    # freeway & Enduro sigma = 3 lr_mean = 0.2 lr_sigma = 0.1 eva = 5 mu = 15 lam = 5 phi = 0.001 fixed
    # breakout 同上 phi = 0.0001 eva = 3 sigma_init = 3 mu = 15

    tune_params(
        namemark,
        ncpu,
        envs_list,
        mu,
        lam,
        sigma_init,
        lr_mean,
        lr_sigma,
        phi,
    )

    
