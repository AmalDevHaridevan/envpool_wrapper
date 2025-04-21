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

from envpool_protocol.envpool_protocol import EnvPoolProtocol
import sys
if sys.version_info.minor > 10:
    raise RuntimeError("Python version >= 3.11 is not supported")
import inspect
import gym
import os
import subprocess

class EnvManager:
    managed_envs_ = list()
    
    def register_env(env):
        if isinstance(env, type):
            if not issubclass(env, (EnvPoolProtocol,)):
                raise RuntimeError(f"env does not subclass EnvPoolProtocol")
            if not issubclass(env, (gym.Env,)):
                raise RuntimeError(f"env does not subclass gym.Env")
            if not (env.__init__.__code__.co_argcount == 2 and "envid" in env.__init__.__code__.co_varnames):
                raise RuntimeError(f"env __init__ should take envid arg as keyword arg")
        elif not isinstance(env, EnvPoolProtocol):
            raise RuntimeError(f"env does not subclass EnvPoolProtocol")
        elif not isinstance(env, gym.Env):
            raise RuntimeError(f"env does not subclass gym.Env")
        if isinstance(env, type):
            EnvManager.managed_envs_.append(env)
        else:
            if not (type(env).__init__.__code__.co_argcount == 2 and "envid" in type(env).__init__.__code__.co_varnames):
                raise RuntimeError(f"env __init__ should take envid arg as keyword arg")
            EnvManager.managed_envs_.append(type(env))
    
class CppRegisterWrapper:
    cpp_envs_ = list()
    
    def register(module_pth, envname, scope, **kwargs):
        import envpool_wrap
        from envpool_wrap.registration import register
        import numpy
        if numpy.__version__.split(".")[0] != "1":
            raise RuntimeError("Numpy version major not supported, only 1 is supported")
        locals().update(scope)
        register(
        task_id=envname,
        import_path=module_pth,
        **kwargs
        )

class CppWrapperGenerator:
    def generate_envpool_wrap(cls, verbose=False):
        """Generates envpool wrapper for any registered class"""
        if not isinstance(cls, type):
            raise RuntimeError(f"Expected cls to be an instance but got {cls}")
        if cls not in EnvManager.managed_envs_:
            EnvManager.register_env(cls)
        # extract useful stuff
        clsname = cls.__name__
        try:
            clsdef_file = inspect.getfile(cls)
            clsdef_pth, clsdef_file = os.path.split(clsdef_file)
            clsdef_file, _ = os.path.splitext(clsdef_file)
        except (OSError, TypeError):
            print(f"Could not detect source file for class definition of class {cls}, wrapping failed")
            return
        cls_obj = cls(0)
        observation_space = cls_obj.observation_space
        action_space = cls_obj.action_space
        del cls_obj
        # defs
        if len(action_space.shape) != 1:
            raise RuntimeError(f"Expected action space to be 1-D, but got {len(action_space.shape)} dims")
        if len(observation_space.shape) != 1:
            raise RuntimeError(f"Expected observation space to be 1-D, but got {len(observation_space.shape)} dims")
        
        defs = dict(ACTION_SPACE_DIM = action_space.shape[0],
                    OBSERVATION_SPACE_DIM = observation_space.shape[0],
                    MODULE_NAME = f"{clsname}_wrap",
                    OBSERVATION_SPACE_LOW = observation_space.low.tolist().__repr__().replace('[','{').replace(']','}'),
                    OBSERVATION_SPACE_HIGH = observation_space.high.tolist().__repr__().replace('[','{').replace(']','}'),
                    ACTION_SPACE_LOW = action_space.low.tolist().__repr__().replace('[','{').replace(']','}'),
                    ACTION_SPACE_HIGH = action_space.high.tolist().__repr__().replace('[','{').replace(']','}'),
                    RUNTIME_MODULE = clsdef_file,
                    RUNTIME_MODULE_PATH = clsdef_pth,
                    PROTOCOL_CLS=clsname,
                    SPEC_CLS=f"{clsname}_wrapSpec",
                    POOL_CLS=f"{clsname}_wrapPool"
                    )
        # determine envpool wrap dir
        base_dir, _ = os.path.split(__file__)
        root_dir, _ = os.path.split(base_dir)
        envpool_wrap_dir = os.path.join(root_dir, "envpool_wrap")
        envpool_lib_dir = os.path.join(envpool_wrap_dir, "env_libs")
        CppWrapperGenerator.generate_pylib(defs=defs, root_dir=root_dir, install_dir=envpool_lib_dir, verbose=verbose)
        
    def generate_pylib(defs, root_dir, install_dir, verbose):
        build_dir = "/tmp/_envpool"
        try:
            os.makedirs(build_dir)
        except (FileExistsError, OSError):
            try:
                subprocess.check_output(f"rm -rf {build_dir}".split(" "))
            except (subprocess.CalledProcessError, FileNotFoundError):
                ...
            os.makedirs(build_dir)
        try:
            subprocess.check_output(f"cp {root_dir}/. {build_dir}/ -r".split(" "))
        except (subprocess.CalledProcessError):
            print(f"could not copy files to {build_dir}, wrapping failed")
            return
        def_file = os.path.join(build_dir, "envpool_protocol", "include","definitions.hh")
        CppWrapperGenerator.write_defs(defs=defs, def_file=def_file)
        cmake_config_cmd = f"cmake -S {build_dir}/envpool_protocol -B {build_dir}/build -DCMAKE_INSTALL_PREFIX:STRING=\{install_dir}"
        cmake_cmd = f"cmake --build {build_dir}/build --target install "
        try:
            subprocess.check_output(cmake_config_cmd.split(" "))
        except (subprocess.CalledProcessError):
            print(f"could not finish cmake configuration stage for wrapper, wrapping failed")
            return
        try:
            subprocess.check_output(cmake_cmd.split(" "), stderr=subprocess.DEVNULL if not verbose else None)
        except (subprocess.CalledProcessError):
            print(f"could not finish build process for wrapper, wrapping failed")
            return
        print("")
        
    def write_defs(defs, def_file):
        with open(def_file, "w") as file:
            RUNTIME_MODULE = defs.pop("RUNTIME_MODULE")
            RUNTIME_MODULE_PATH = defs.pop("RUNTIME_MODULE_PATH")
            PROTOCOL_CLS = defs.pop("PROTOCOL_CLS")
            file.write(f"#define RUNTIME_MODULE \"{RUNTIME_MODULE}\"\n")
            file.write(f"#define RUNTIME_MODULE_PATH \"{RUNTIME_MODULE_PATH}\"\n") 
            file.write(f"#define PROTOCOL_CLS \"{PROTOCOL_CLS}\"\n")    
            for key in defs:
                line = f"#define {key} {defs[key]}\n"
                file.write(line)
            
        