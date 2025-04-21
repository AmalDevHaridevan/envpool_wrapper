# envpool_wrapper
Envpool wrapper for Envs written in Python. This module defines a protocol, allowing to wrap aribitrary environments written in Python to be executed using envpool API.

# Envpool
To learn about Envpool, see https://github.com/sail-sg/envpool

# Benefits of Envpool
Envpool allows fast multithreaded execution of environments. This is typically faster than parallelization using multiple processes, and allows to fully utilize cpu resources.

# Benchmark
TODO
# Minimal Example
```python
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
    # or rerunning the logic when imported in embedded envpool code
    EnvManager.register_env(MyEnv)
    CppWrapperGenerator.generate_envpool_wrap(MyEnv, verbose=True)

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
