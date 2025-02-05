from gymnasium_env.envs.grid_world import GridWorldEnv
from gymnasium_env.envs.CliffWalker import CliffWalker
from gynmasium.envs.registration import register

register(
	id="gymnasium_env/CliffWalker-v0",
	entry_point="gymnasium_env.cliffwalker:CliffWalker",
)
