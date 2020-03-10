import torch.multiprocessing as mp
import time


# TODO: Test the apply_async and multiprocess
# def job(x):
#     time.sleep(3-x)
#     return x


# if __name__ == "__main__":
#     start = time.time()
#     pool = mp.Pool()
#     res = [pool.apply_async(job,(i,)) for i in range(3)]
#     for r in res:
#         print (r.get())
#     end = time.time()
#     print(end-start)
# 0
# 1
# 2


# TODO: Test get reward_atari and test() (don't parallel)
# if __name__ == "__main__":
#     # distribute training in parallel
#     from main import ARGS
#     from model import build_model
#     from train import get_reward_atari
#     from vbn import explore_for_vbn

    
#     import gym,torch
#     env = gym.make("AlienNoFrameskip-v4")
#     ARGS.action_n = env.action_space.n
#     model = build_model(ARGS)
#     # vitural batch normalization
#     refer_batch_torch = None
#     if ARGS.env_type == "atari":
#         # get reference batch
#         reference_batch = explore_for_vbn(env, 0.01, ARGS)
#         refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
#         for i in range(ARGS.refer_batch_size):
#             refer_batch_torch[i] = reference_batch[i]
    
#     result = get_reward_atari(model, None, None, env, ARGS, refer_batch_torch, None, True, False)
#     print(result)

#TODO: Test optimizer update div and fisher , f 
# #TODO: Test optimizer log and update gauss
# if __name__ == "__main__":
#     from main import ARGS,build_mean,build_sigma
#     from model import build_model
#     from train import get_reward_atari
#     from vbn import explore_for_vbn

#     import gym,torch
#     from optimizer import optimize,log_info,update_info

#     env = gym.make("AlienNoFrameskip-v4")
#     ARGS.action_n = env.action_space.n
#     ARGS.lam = 2
#     ARGS.population_size = 3
#     ARGS.sigma_init = 1
#     ARGS.phi = 0.0001
#     ARGS.lr_mean = 0.2
#     ARGS.lr_sigma = 0.001
#     model_list = [build_model(ARGS) for i in range(ARGS.population_size)]

#     model = model_list[0]
#     mean_list = [build_mean(model, ARGS) for i in range(ARGS.lam)]
#     sigma_list = [build_sigma(model, ARGS) for i in range(ARGS.lam)]

#     mean = mean_list[0]
#     sigma = sigma_list[0]

#     noise = {}
#     for model in model_list:
#         for name,params in model.named_parameters():
#             tmp = model
#             for attr_value in name.split("."):
#                 tmp = getattr(tmp, attr_value)
#             tmp.add_(mean[name])
#             noise[name] = torch.randn_like(tmp,dtype=torch.float) * sigma[name]
#             tmp.add_(noise[name])
  
#     #rewards = [[140.0,260.0,1000.0],[1200,1100,490]]
#     rewards = [[0,0,0],[0,0,0]]
    
#     optimize(0,mean,sigma,model_list,mean_list,sigma_list,rewards[0],ARGS)
#     # import numpy as np
    # import os

    # mean_opt.update_gauss(mean,sigma,ARGS)
    # info = mean_opt.update_info(mean)
    # print(info)
    # mean_opt.log_info(mean,sigma,0,os.getcwd())

    # utility=[-1.01010,0.1112,1.23342]
    # mean_opt.zero_ffd()
    # mean_opt.log_info(mean,sigma,0,os.getcwd())
    # mean_opt.update_fit_fisher(mean,sigma,noise_list[0],utility,ARGS.population_size)
    # mean_opt.log_info(mean,sigma,0,os.getcwd())
    # mean_opt.update_div(0,mean,sigma,mean_list,sigma_list)
    # mean_opt.log_info(mean,sigma,0,os.getcwd())
    # mean_opt.update_gauss(mean,sigma,ARGS)
    # mean_opt.log_info(mean,sigma,0,os.getcwd())
    # info = mean_opt.update_info(mean)
    # print(info)
    # mean_opt.log_info(mean,sigma,0,os.getcwd())

    # for i in range(ARGS.lam):
    #     mean = mean_list[i]
    #     sigma = sigma_list[i]
    #     mean_opt.optimize(i,mean,sigma,noise_list[i],mean_list,sigma_list,rewards[i],ARGS)
    #     sigma_opt.optimize(i,mean,sigma,noise_list[i],mean_list,sigma_list,rewards[i],ARGS)



# import os
# import logging
# from util import setup_logging
# if __name__ == "__main__":

#     fold = os.getcwd()
#     for i in range(3):
#         logger = logging.getLogger(__name__)
#         filename = str(i)+".log"
#         logger = setup_logging(logger,fold,filename)
#         logger.info(str(i))
#         logger.info(str(i))

# # TODO:test check bound
# import torch

# A = torch.Tensor([-11,2,11])
# H = 10
# L = -10
# B = torch.ones_like(A) * 2 * H - A
# C = torch.ones_like(A) * 2 * L - A
# D = torch.min(A,B)
# D = torch.max(D,C)
# print(D)


#TODO: Test the check bound is work?
# import torch
# A = torch.Tensor([[-11,2,11],[-11,2,11]])
# from optimizer import check_bound
# B = {}
# B['a']=A
# B['a1']=A
# def updategauss(B):
#     for name,params in B.items():
#         B[name]=check_bound(params,10,-10)
# def log(B):
#     C = []
#     for name,params in B.items():
#         C.append(params.view(-1))
#     C = torch.cat(C).float()
#     print(C)
#     print(C.max())
#     print(C.min())
#     print(C.mean())
#     print(C.var())

# updategauss(B)
# log(B)
#C = check_bound(C,10,-10)
# B = torch.Tensor([-11,2,11])
# C = torch.Tensor([-11,2,11])
# p = [A.view(-1),B,C]
# D = torch.cat(p)

# import torch
# from optimizer import check_bound
# A = torch.Tensor([[-0.000001,0.00001,0.0000001],[-0.0000002,0.0000002,0.0000001]])
# B = torch.Tensor([[1,1],[1,1]])
# C = torch.Tensor([[2,3],[4,5]])
# M = {
#     'a':torch.Tensor([[3,4],[5,6]]),
#     'b':torch.Tensor([[2,3],[4,5]])
# }
# M_L = [M,M,M]
# f = {}
# ulity = 1
# for i,M in enumerate(M_L):
#     for name,params in M.items():
#         if name not in f.keys():
#             f[name] = torch.zeros_like(M[name],dtype=torch.float)
        
#         D = 1/B
#         E = C - M[name]
#         f[name].add_(D*E*ulity)
#         print(f[name])

# print(A * A * A *  A)
# tmp = torch.Tensor([[-1,1,1],[-2,1,2]])
# print(tmp - A)

# import torch

# A = torch.Tensor([[[2,4],
#                   [1,2]],
#                   [[2,4],
#                   [1,2]]])
# sigma = torch.ones_like(A) * 1
# B = torch.Tensor([[2,4],
#                   [1,2]])
# At = A.permute(2,1,0)
# print(At)
# sigma_B = torch.ones_like(B)
# Bt = torch.transpose(B,0,1)
# sigma_in = 1 / sigma
# print(sigma_in)
# fmean = sigma_in.mm(sigma_in.mm(B.mm(Bt)))
# print(fmean)
# fmean2 = sigma_in * B * B * sigma_in
# print(fmean2)
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from main import build_mean,build_sigma,ARGS
from train import train,test
from optimizer import check_bound,optimize_parallel,optimize
from vbn import explore_for_vbn
import numpy as np
import time
import gym
from model import build_model

class Net(nn.Module):
    def __init__(self, ARGS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5,5)
        self.fc2 = nn.Linear(5,5)
        self.set_parameter_no_grad()
        self._initialize_weights()

    
    def _initialize_weights(self):
        for m in self.modules():
            # Orthogonal initialization and layer scaling
            # Paper name : Implementation Matters in Deep Policy Gradient: A case study on PPO and TRPO
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_() 

    def set_parameter_no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_size(self):
        """
        Returns:   
            Number of all params
        """
        count = 0
        for params in self.parameters():
            count += params.numel()
        return count
    
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = out.view(-1,out.size(0))
        return F.softmax(out, dim=1)

def build_model(ARGS):
    """
    Args:
        action_n(int):    Available action numbers of one game
    Returns:   
        ESNet(nn.Module): Neural network module with specified architecture
    """
    return Net(ARGS)


def add_noise(model,mean,sigma):

    with torch.no_grad():
        for name,params in model.named_parameters():
            A = torch.randn_like(params,dtype=torch.float) * sigma[name] + mean[name]
            params.data = torch.clamp(A,ARGS.L,ARGS.H)

def reward(reward,lam,mu):
    for i in range(lam):
        for j in range(mu):
            reward[i][j]+=np.random.randint(0,20)*10
    return reward

if __name__ == "__main__":

    # set up multiprocessing
    mp.set_sharing_strategy("file_system")
    # log and save path setting
    torch.set_num_threads(1)
    # torch.manual_seed(int(time.time()))

    ARGS.lam = 4
    ARGS.population_size = 14
    ARGS.sigma_init = 1
    ARGS.phi = 0.0001
    ARGS.lr_mean = 0.2
    ARGS.lr_sigma = 0.01
    ARGS.ncpu = 85
    ARGS.eva_times = 3

    ARGS.gamename = "AlienNoFrameskip-v4"
    env = gym.make(ARGS.gamename)
    env.seed(int(time.time()))
    ARGS.action_n = env.action_space.n
    
    refer_batch_torch = None
    # get reference batch
    reference_batch = explore_for_vbn(env, 0.01, ARGS)
    refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
    for i in range(ARGS.refer_batch_size):
        refer_batch_torch[i] = reference_batch[i] 
    
    model = build_model(ARGS)
    mean_list = [build_mean(model, ARGS) for i in range(ARGS.lam)]
    sigma_list = [build_sigma(model, ARGS) for i in range(ARGS.lam)]

    pool = mp.Pool(processes=ARGS.ncpu)
    rewards_list = np.random.randint(0,200,size = (ARGS.lam,ARGS.population_size))

    for i in range(10):
        # for idx,models in enumerate(model_list):
        #     for j,model in enumerate(models):
        #         add_noise(model,mean_list[idx],sigma_list[idx])
        
        rewards_list, frame_count, models_list, noops_list, detail_rewards = train(
            mean_list,
            sigma_list,
            pool,env,
            ARGS,
            refer_batch_torch
        )
        # rewards_list = reward(rewards_list,ARGS.lam,ARGS.population_size)
        optimize_parallel(i,mean_list,sigma_list,models_list,rewards_list,pool,ARGS)

        print(mean)
        print(sigma)
