import gym
import torch
import torch.multiprocessing as mp
from gym import wrappers, logger

from src.train import test,get_reward_atari
from src.preprocess import ProcessUnit
from src.main import ARGS
from src.model import build_model
from src.vbn import explore_for_vbn
from src.preprocess import ProcessUnit

def viz(gamename, model_path, no_op, ARGS):
    ARGS.gamename = gamename
    ARGS.ncpu = 11
    seed = 111
    env = gym.make(ARGS.gamename+'NoFrameskip-v4')
    outdir = './video/'+ ARGS.gamename+'/'+str(no_op)
    env = wrappers.Monitor(env, directory=outdir, force=True)

    ARGS.action_n = env.action_space.n
    ProcessU = ProcessUnit()
    env.seed(seed)

    model = build_model(ARGS)
    model.load_state_dict(torch.load(model_path))

    #reference_batch = explore_for_vbn(env, 0.01, ARGS)
    #refer_batch_torch = torch.zeros((ARGS.refer_batch_size, 4, 84, 84))
    #for i in range(ARGS.refer_batch_size):
    #    refer_batch_torch[i] = reference_batch[i]

    #pool = mp.Pool(processes=ARGS.ncpu)
    #result = get_reward_atari(model,None, None,env, seed, ARGS, refer_batch_torch ,None,True,True)
    reward = 0
    break_is = False
    
    model.switch_to_vbn()
    ob = env.reset()
    ProcessU.step(ob)
    
    for i in range(no_op):
        # 0 is Null Action but have not found any article about the meaning of every actions
        observation, reward, done, _ = env.step(0)
        ProcessU.step(observation)

    for _ in range(10000000):
        action = model(ProcessU.to_torch_tensor())[0].argmax().item()
        for i in range(4):
            ob, reward, done, __ = env.step(action)
            ProcessU.step(ob)
            env.render('rgb_array')
            if done:
                break_is = True
                break
        if break_is is True:
            break
    env.close()

if __name__ == "__main__":
    viz('Freeway','./model/Freeway-phi-0.0001-lam-5-mu-1581.pt',6,ARGS)
    #viz('Enduro','./model/Enduro-phi-0.001-lam-5-mu-1520.pt',6,ARGS)
    #viz('BeamRider','./model/BeamRider-phi-1e-05-lam-5-mu-159.pt',0,ARGS)
