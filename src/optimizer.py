#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   optimizer.py
@Time    :   2020/02/01 17:07:55
@Describtion: Optimizer of mean and sigma of Guass distributions    
"""

# here put the import lib

import torch
import os
import numpy as np
import copy

def check_zero(tensor):
    t = tensor.numpy()
    have_zero = np.any(t <= 0.00000001)
    return have_zero

def check_bound(A,H,L):
    """check bound of grad * lr.  
    Args:
        A:      Tensor to be checked.If grad * lr is out of range(L,H), delta = 2H-delta or 2L-delta.
        H:      High value.  
        L:      Low value.  
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(
        "the gauss distribution parameters must be a tensor, Got{}".format(torch.typename(A))
    )
    B = torch.ones_like(A,dtype=torch.float) * 2 * H - A
    C = torch.ones_like(A,dtype=torch.float) * 2 * L - A
    A = torch.min(A,B)
    A = torch.max(A,C)
    A = torch.clamp(A,L,H)
    return A

def update_info(params):
    if not isinstance(params, dict):
        raise TypeError(
        "the parameters must be a dict with key(name)"
        "value(torch.tensor), Got{}".format(torch.typename(params))
    )
    params_list = []
    for name, value in params.items():
        params_list.append(value.view(-1))
    params_t = torch.cat(params_list).float()
    pmax = torch.max(params_t)
    pmin = torch.min(params_t)
    pmean = torch.mean(params_t)
    pvar = torch.var(params_t)
    return pmax, pmin, pmean, pvar

def log_info(gen,mean,sigma,idx,fitm,fits,fisherm,fishers,divm,divs,gradm,grads,folder_path):
    with open(os.path.join(folder_path,"state.txt"), "a") as f:
        f.write("=======Gen %3d Model %2d ========\n" % (gen,idx))
        f.write("  sigma: ")
        info = update_info(sigma)
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        f.write(" mean  : ")
        info = update_info(mean)
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(fitm)
        f.write(" fit mean: ")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(fits)
        f.write(" fit sigma: ")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(fisherm)
        f.write(" fisher mean: ")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(fishers)
        f.write(" fisher sigma: ")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(divm)
        f.write(" div mean  :")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(divs)
        f.write(" div sigma  :")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(gradm)
        f.write(" grad mean :")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))
        info = update_info(grads)
        f.write(" grad sigma :")
        f.write("%.8f, %.8f, %.8f, %.8f\n" % (info[0],info[1],info[2],info[3]))

def optimize(gen,idx,mean, sigma, model_list, mean_list, sigma_list, rewards,ARGS):
    # scale_rewards = reward_scale(rewards)
    rank = [0 for i in range(len(rewards))]
    for r,i in enumerate(np.argsort(rewards)[::-1]):
        rank[i] = r + 1  # rank kid by reward
    mu = ARGS.population_size
    util_ = np.maximum(0, np.log(mu / 2 + 1) - np.log(rank))
    utility = (util_ / (util_.sum())) - (1 / mu)

    fmean = {}
    fsigma = {}
    fishermean = {}
    fishersigma = {}
    for name, params in model_list[0].named_parameters():
        if name not in fmean.keys():
            fmean[name] = torch.zeros_like(params, dtype=torch.float)
            fsigma[name] = torch.zeros_like(params, dtype=torch.float)
            fishermean[name] = torch.zeros_like(params, dtype=torch.float)
            fishersigma[name] = torch.zeros_like(params, dtype=torch.float)
                
    for k, model in enumerate(model_list): 
        for name, params in model.named_parameters():
            noise = params.data - mean[name]

            sigma_inverse = 1 / sigma[name]
            tmp1 = sigma_inverse*noise*noise*sigma_inverse
            tmp1 = torch.clamp(tmp1,0.0001,1000)
            tmp_ = tmp1 - sigma_inverse

            fmean[name].add_(sigma_inverse*noise*utility[k])
            fsigma[name].add_(tmp_*utility[k])
            fishermean[name].add_(tmp1)
            fishersigma[name].add_(torch.clamp(tmp_*tmp_,0.0001,1000))

            # if check_zero(fishermean[name]):
            #     print("fishermean<1e-8")
            #     with open('error.txt','a') as f:
            #         f.write( name+'\n')
            #         f.write( 'fmisherean'+str(fishermean[name])+'\n')
            #         f.write('noise'+str(noise)+'\n')
            #         f.write('tmp1'+str(tmp1)+'\n')

    for name in fmean.keys():
        fmean[name].mul_(1/ARGS.population_size)
        fsigma[name].mul_(1/2/ARGS.population_size)
        fishermean[name].mul_(1/ARGS.population_size)
        fishersigma[name].mul_(1/4/ARGS.population_size)
    
    dmean = {}
    dsigma = {}
    for name,params in mean.items():
            dmean[name] = torch.zeros_like(params, dtype=torch.float)
            dsigma[name] = torch.zeros_like(params, dtype=torch.float)
    for mean2, sigma2 in zip(mean_list, sigma_list):
        for (name1, params1),(name2, params2) in zip(mean.items(),mean2.items()):
            sigma_part = 2/(sigma[name1]+sigma2[name1])
            params_minus = params1 - params2
            dmean[name1].add_(sigma_part*params_minus)
            dsigma[name1].add_(sigma_part-1/4*sigma_part*params_minus*params_minus*sigma_part-1/sigma[name1])
    # with open('temp.txt','a')as f:
    #     f.write('dmean'+str(dmean)+'\n')
    #     f.write('dsigma'+str(dsigma)+'\n')

    for name in dmean.keys():
        dmean[name].mul_(1/4)
        dsigma[name].mul_(1/4)

    mean_grad = {}
    sigma_grad = {}
    for name in dmean.keys():
        mean_grad[name] = 1/fishermean[name]*(fmean[name]+ARGS.phi*dmean[name])
        sigma_grad[name] = 1/fishersigma[name]*(fsigma[name]+ARGS.phi*dsigma[name])
    # with open('temp.txt','a')as f:
    #     f.write('meangrad'+str(mean_grad)+'\n')
    #     f.write('sigmagrad'+str(sigma_grad)+'\n')


    for name, params in sigma.items():
        sigma[name].add_(ARGS.lr_sigma * sigma_grad[name])
        sigma[name] = torch.clamp(sigma[name],1e-8,1e8)

    for name, params in mean.items():
        mean[name].add_(ARGS.lr_mean * mean_grad[name])    
        mean[name] = torch.clamp(mean[name],ARGS.L,ARGS.H)

    log_info(gen,mean,sigma,idx,fmean,fsigma,fishermean,fishersigma,dmean,dsigma,mean_grad,sigma_grad,ARGS.folder_path)
    return True

# def optimize_replaced(gen,idx,mean, sigma, model_list, mean_list, sigma_list, rewards,ARGS):
#     rank = np.argsort(rewards)[::-1] + 1 # rank kid by reward

def optimize_parallel(gen,mean_list,sigma_list,model_list,rewards,pool,ARGS):
    save_mean_list = copy.deepcopy(mean_list)
    save_sigma_list = copy.deepcopy(sigma_list)
    jobs = [
        pool.apply_async(
        optimize,
        (gen,i,mean_list[i],sigma_list[i],model_list[i],save_mean_list,save_sigma_list,rewards[i],ARGS)
        ) for i in range(ARGS.lam)
    ]   

    done = []
    for job in jobs:
        done.append(job.get())
    #for i in range(ARGS.lam):
    #    mean = mean_list[i]
    #    sigma = sigma_list[i]
    #    optimize(gen,i,mean,sigma,model_list[i],save_mean_list,save_sigma_list,rewards[i],ARGS)

def optimize_serial(gen,mean_list,sigma_list,model_list,rewards,ARGS):
    save_mean_list = copy.deepcopy(mean_list)
    save_sigma_list = copy.deepcopy(sigma_list)
    for i in range(ARGS.lam):
        optimize(gen,i,mean_list[i],sigma_list[i],model_list[i],save_mean_list,save_sigma_list,rewards[i],ARGS)
