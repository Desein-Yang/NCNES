from src.model import ESNet,build_model
from main import ARGS
import gym

# init rl environment 
env = gym.make(ARGS.gamename)

ARGS.env_type = "atari"
# init ARGS's parameter
ARGS.action_n = env.action_space.n

model = build_model(ARGS)
model.set_parameter_no_grad()

