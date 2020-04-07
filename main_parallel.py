#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main_parallel.py
@Time    :   2020/02/01 17:58:21
@Describtion:   main function to run experiment(population_size in parallel)
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

from src.optimizer import optimize_parallel
from src.train import train_parallel,test
from src.model import build_model, build_mean,build_sigma
from src.util import mk_folder, save, load, setup_logging
from src.vbn import explore_for_vbn

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
    eva_time_decay = True
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
        logger.info("evatime decay enable ?:%s" % cls.eva_time_decay)
        logger.info("H: %s; L: %s" % (cls.L, cls.H))
        logger.info("EvaluateTimes %s" % cls.eva_times)

    @classmethod
    def set_params(cls, env, kwargs):
        """Set up hyperparameters in ARGS class"""
        # cls.eva_times = kwargs["eva_times"]

        gamename = cls.gamename.split('N')[0]
        cls.action_n = env.action_space.n
        cls.checkpoint_name = gamename+"-phi-" + str(cls.phi) + "-lam-" + str(cls.lam) + "-mu-" + str(cls.population_size)
        
        if gamename in ['Alien','Qbert','SpaceInvaders','BeamRider']:
            cls.eva_times = 5
        elif gamename in ['Breakout','Seaquest']:
            cls.eva_times = 1
        else:
            cls.eva_times = 3
        if gamename in ['Freeway','Enduro']:
            cls.phi = 0.001
        elif gamename in ['BeamRider','SpaceInvaders']:
            cls.phi = 0.00001
        
        cls.phi = kwargs["phi"]
        cls.lam = kwargs["lam"]
        cls.population_size = kwargs["mu"]
        cls.lr_mean = kwargs["lr_mean"]
        cls.sigma_init = kwargs["sigma_init"]
        cls.lr_sigma = kwargs["lr_sigma"]
        cls.timestep_limit = kwargs["frame"]

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger
    
    @classmethod
    def set_gamename(cls, gamename):
        if cls.env_type == "atari":
            cls.gamename = "%sNoFrameskip-v4" % gamename
            #cls.gamename = "%sDeterministic-v4" % gamename
            
    @classmethod
    def set_folder_path(cls,folder_path):
        cls.folder_path = folder_path
    
    @classmethod
    def set_logfile_name(cls,logfile_name):
        cls.logfile_name = cls.gamename.split('N')[0] + cls.namemark + "-phi-" + str(cls.phi) + "-lam-" + str(cls.lam) + "-mu-" + str(cls.population_size)+".txt"

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
    # np.random.seed(111)
    # seed = np.random.randint(0, 1000000)
    seed = [np.random.randint(1,1000000) for i in range(ARGS.population_size)]
    logger.info("seed:%s", str(seed))
    # env.seed(123456)

    # init best and timestep
    best_test_score = ARGS.Small_value
    best_train_score = ARGS.Small_value
    timestep_count = 0
    model_best = build_model(ARGS)
    model_best.set_parameter_no_grad()
    model_size = model_best.get_size()
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

        if ARGS.eva_time_decay:
            ARGS.eva_times = int(ARGS.eva_times * percent)
            if ARGS.eva_times < 1:
                ARGS.eva_times = 1
            if ARGS.eva_times > 10:
                ARGS.eva_times = 10

        # sample and evaluate
        rewards_list, frame_count, models_list, noops_list, detail_rewards = train_parallel(
            mean_list,
            sigma_list,
            pool,env,
            ARGS,
            refer_batch_torch,
            seed
        )
        timestep_count += frame_count
        rewardlist_mean = [np.mean(rewards_list[i]) for i in range(ARGS.lam)]
        rewardlist_var = [np.var(rewards_list[i]) for i in range(ARGS.lam)]
        logger.info("=============================================")
        logger.info("Gen            :%2d " % g)
        logger.info("Framecount     :%9d " % (frame_count))
        logger.info("AllFramecount  :%s/%s" % (timestep_count,ARGS.timestep_limit))
        logger.info("Rewardlist     :%s " % str(rewards_list))
        logger.info("Noops list     :%s " % str(noops_list))
        logger.info("Rewardlist mean:%s " % str(rewardlist_mean))
        logger.info("Rewardlist var :%s " % str(rewardlist_var))
        logger.info("DetailReward   :%s " % str(detail_rewards))
  
        # save best one model
        index = np.array(rewards_list).argmax()
        best_model_i, best_model_j = (
            index // ARGS.population_size,
            index % ARGS.population_size,
        )

        # calculate gradient and update distribution in parallel
        optimize_parallel(g,mean_list,sigma_list,models_list,rewards_list,pool,ARGS)

        # check timestep
        if timestep_count > ARGS.timestep_limit:
            logger.info("Satisfied timestep limit")
            break
    
    pool.close()
    pool.join()
    savepath = save(model_best, ARGS.checkpoint_name, ARGS.folder_path,g)


@click.command()
@click.option('--namemark', default='final')
@click.option('--ncpu', default=40)
@click.option('--sigma_init', default=2)
@click.option('--phi', default=0.00001)
@click.option('--lr_mean',default=0.2)
@click.option('--lr_sigma',default=0.1)
@click.option('--lam',default=5)
@click.option('--mu',default=15)
@click.option('--game',default='Freeway')
@click.option('--frame',default=1e6)
def run(
    namemark,
    ncpu,
    mu,
    lam,
    sigma_init,
    lr_mean,
    lr_sigma,
    phi,
    game,
    frame
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
            "frame": frame
        }
    )
    
    # set folder path
    folder_path = mk_folder(os.path.join(os.getcwd(), "parallel_log"))
    print("start!")

    idx = 0
    print("-----------------------------------------")
    print("the number of hyperparameters combination:", len(kwargs_list))

    # set logger handler and run main
    logger = logging.getLogger(__name__)
    for params in kwargs_list:
        ARGS.set_gamename(game)
        ARGS.set_folder_path(folder_path)
        logfile = (namemark + "-" + game + "-phi-" + str(params["phi"])+'.txt')
        logger = setup_logging(logger, folder_path, logfile)
        main(ARGS, logger, params)
        print("finish idx %s : %s for game:%s" % (idx, str(params), game))
        idx += 1

if __name__ == "__main__":
    run()
