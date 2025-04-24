# envpool_wrapper
Envpool wrapper for Envs written in Python. This module defines a protocol, allowing to wrap aribitrary environments written in Python to be executed using ```envpool``` API. 
Wrapped environments can be used just like ```gym``` envs, or in a similar manner to ```envpool``` envs. 
This wrapping makes your environment comaptible with ```torchrl``` ```MultiThreadedEnvWrapper```. https://pytorch.org/rl/main/reference/generated/torchrl.envs.MultiThreadedEnv.html

# Envpool
To learn about Envpool, see https://github.com/sail-sg/envpool

# Benefits of Envpool
Envpool allows fast multithreaded execution of environments. This is typically faster than parallelization using multiple processes, and allows to fully utilize cpu resources. Envpool allows to massively parllelize Deep Reinforcement Learning training.

# Benchmark
TODO
# Minimal Example
```python
from envpool_protocol.envpool_protocol import EnvPoolProtocol
from envpool_protocol.manager import EnvManager, CppWrapperGenerator
import gym

class MyEnv(EnvPoolProtocol, gym.Env):
    def __init__(self, envid):
        # init should only take a single argument , the envid
        # any other definitions will throw error, and cannot be registered
        # this is to keep the protocol simple
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    
    def _step_envpool(self, act):
        # do state transitions, etc...
        # store it in self if needed for later reference
        # should return nothing
        print("got act ", act)
    
    def _reward_envpool(self):
        # this is called after _make_obs_envpool
        # returns a scalar
        return 2.0
    
    def _reset_envpool(self):
        # reset internal sim state or external interface states
        # should return nothing
        ...
    
    def _make_obs_envpool(self):
        # this is called after _step_envpool and _reset_envpool
        # you should generate any states from internal simulation or
        # external interfaces, and store for future reference
        # returns numpy array
        obs = self.observation_space.sample()
        print("sending obs: ", obs)
        return obs 
    
    def _done_envpool(self):
        # return if the done | truncated condition has reached
        return False

if __name__ == "__main__":
    # essential to wrap it in main, or else it will cause circular imports
    # or rerunning the logic when imported in embedded envpool code
    EnvManager.register_env(MyEnv)
    CppWrapperGenerator.generate_envpool_wrap(MyEnv)

```
## In a second file
```python
import envpool_wrap as envpool
print(envpool.list_all_envs()) # this should list an env named MyEnv_wrap
env = envpool.make(task_id="MyEnv_wrap", env_type="gym")
obs = env.reset()
import numpy as np
act= np.zeros((1,1)) +1
env.step(act)
```
# Requirements
Currently you need to have ```numpy``` major version to be less than 2. 
```Python``` minor shall be less than ```11``` (non inclusive)
# Docker
We provide a dockerfile for reproducing results. See ```docker/Dockerfile```
