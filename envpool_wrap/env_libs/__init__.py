#    Copyright 2025 Amaldev Haridevan

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy
if numpy.__version__.split(".")[0] != "1":
  raise RuntimeError("Numpy version major not supproted, only 1 is supported")
import glob
import os
from envpool_wrap.python.api import py_env
from envpool_wrap.python.dm_envpool import DMEnvPoolMeta
from envpool_wrap.python.env_spec import EnvSpecMeta
from envpool_wrap.python.gym_envpool import GymEnvPoolMeta
from envpool_wrap.python.gymnasium_envpool import GymnasiumEnvPoolMeta
from envpool_wrap.python.protocol import EnvPool, EnvSpec
import importlib
# register all known envs
__all__ = []

base_dir, _ = os.path.split(__file__)
libs = glob.glob(f"{base_dir}/*cpython*.so")
for lib in libs:
    modname = os.path.split(lib)[1].split(".")[0]
    module = importlib.import_module(f"envpool_wrap.env_libs.{modname}")
    envspec, envpool = getattr(module, f"_{modname}Spec"), getattr(module, f"_{modname}Pool")
    ( WrappedEnvSpec,
    WrappedDMEnvPool,
    WrappedGymEnvPool,
    WrappedGymnasiumEnvPool,
    )   = (EnvSpecMeta(f"WrappedEnvSpec_{modname}", (envspec,), {}),  
            DMEnvPoolMeta(f"WrappedDMEnvPool_{modname}", (envpool,), {}),
            GymEnvPoolMeta(f"WrappedGymEnvPool_{modname}", (envpool,), {}),
            GymnasiumEnvPoolMeta(
              f"WrappedGymnasiumEnvPool_{modname}", (envpool,), {}
            ))
    
    # py_env(module._WrappedEnvSpec, module._WrappedEnvPool, f"WrappedEnvSpec_{modname}", f"WrappedEnvPool_{modname}")
    module_specific_vars = {f"WrappedEnvSpec_{modname}": WrappedEnvSpec,
                            f"WrappedDMEnvPool_{modname}": WrappedDMEnvPool,
                            f"WrappedGymEnvPool_{modname}":WrappedGymEnvPool,
                            f"WrappedGymnasiumEnvPool_{modname}":WrappedGymnasiumEnvPool
                            }
    globals().update(module_specific_vars)
    del WrappedEnvSpec
    del WrappedDMEnvPool
    del  WrappedGymEnvPool
    del  WrappedGymnasiumEnvPool
    __all__.extend(list(module_specific_vars.keys()))

from envpool_protocol.manager import CppRegisterWrapper

    
for lib in libs:
    modname = os.path.split(lib)[1].split(".")[0]
    CppRegisterWrapper.register(f"envpool_wrap.env_libs", f"{modname}",
                                    scope=module_specific_vars,
                                    spec_cls=f"WrappedEnvSpec_{modname}",
                                    dm_cls=f"WrappedDMEnvPool_{modname}",
                                    gym_cls=f"WrappedGymEnvPool_{modname}",
                                    gymnasium_cls=f"WrappedGymnasiumEnvPool_{modname}",
                                    
                                    )