
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
    
    """Create wrappers for gym api, for benchmarking"""
    def step(self, act):
        self._step_envpool(act)
        obs = self._make_obs_envpool()
        done = self._done_envpool()
        reward = self._reward_envpool()
        return obs, reward, done, done, {}
        
    def reset(self, *args):
        self._reset_envpool()
        obs = self._make_obs_envpool()
        return obs, {}

# add registration for gym
gym.envs.register(
    id='MyEnv',
    entry_point='example_env:MyEnv',
    max_episode_steps=2000,
)

class MyEnv2(MyEnv):...

if __name__ == "__main__":
    # essential to wrap it in main, or else it will cause circular imports
    # NOTE: All class definitions shall be outside main
    EnvManager.register_env(MyEnv)
    CppWrapperGenerator.generate_envpool_wrap(MyEnv, verbose=False)
    CppWrapperGenerator.generate_envpool_wrap(MyEnv2, verbose=False)
    # the lines below will run with no errors, but during env creation it will cause issues
    # because embedded interpreter won't be able to find class def for MyEnv3
    # NOTE: Below is an example of invalid usage
    class MyEnv3:...
    CppWrapperGenerator.generate_envpool_wrap(MyEnv3, verbose=False)