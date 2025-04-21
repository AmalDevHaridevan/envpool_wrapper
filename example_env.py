
from envpool_protocol.envpool_protocol import EnvPoolProtocol
from envpool_protocol.manager import EnvManager, CppWrapperGenerator
import gym

class MyEnv(EnvPoolProtocol, gym.Env):
    def __init__(self, envid):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    
    def _step_envpool(self, act):
        print("got act ", act)
    
    def _reward_envpool(self):
        return 2.0
    
    def _reset_envpool(self):
        ...
    
    def _make_obs_envpool(self):
        obs = self.observation_space.sample()
        print("sending obs: ", obs)
        return obs 
    
    def _done_envpool(self):
        return False

if __name__ == "__main__":
    # essential to wrap it in main, or else it will cause circular imports
    EnvManager.register_env(MyEnv)
    CppWrapperGenerator.generate_envpool_wrap(MyEnv)
    # breakpoint()