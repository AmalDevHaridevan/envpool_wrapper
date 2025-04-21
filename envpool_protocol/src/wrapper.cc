//  Copyright 2025 Amaldev Haridevan

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.



#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <mutex>
#include <algorithm>
#include "_internal_defs.hh"
#include <stdexcept>
// PYBIND
#include "envpool/core/py_envpool.h"
#include <pybind11/embed.h>

namespace wrapper{

class SPEC_CLS {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict();
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    float fmax = std::numeric_limits<float>::max();
    int observation_dim = OBSERVATION_SPACE_DIM;
    std::vector<float> obs_min OBSERVATION_SPACE_LOW;
    std::vector<float> obs_max OBSERVATION_SPACE_HIGH;
    return MakeDict("obs"_.Bind(Spec<float>({observation_dim}, {obs_min, obs_max})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    int action_dim = ACTION_SPACE_DIM;
    std::vector<float> act_min ACTION_SPACE_LOW;
    std::vector<float> act_max ACTION_SPACE_HIGH;
    return MakeDict("action"_.Bind(Spec<float>({-1, action_dim}, {act_min, act_max})));

  }
};


// using WrappedEnvSpec = EnvSpec<SPEC_CLS>;



class MODULE_NAME : public Env<EnvSpec<SPEC_CLS>>{
    public:
        MODULE_NAME(const Spec& spec, int envid): Env<EnvSpec<SPEC_CLS>>(spec, envid) {
            // see https://github.com/pybind/pybind11/pull/1211
            // we need to release first, and then acquire from a clean state
            // to prevent deadlocks
            pybind11::gil_scoped_release release;
            pybind11::gil_scoped_acquire acquire;
            try {
                py::module_ sys = py::module_::import("sys");
                sys.attr("path").attr("append")(RUNTIME_MODULE_PATH);
                mod = py::module::import(RUNTIME_MODULE);
                env_cls = mod.attr(PROTOCOL_CLS)(envid);
                has_class = true;
            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
                has_class = false;
            }
        };

        void WriteState(float reward, const std::vector<float>& obs){
            State state = Allocate();
            for (int i = 0; i < OBSERVATION_SPACE_DIM; ++i) {
              state["obs"_][i] = obs.at(i);

            }
            state["reward"_] = reward;
        }

        void Reset() override { 
            
            // PyGILState_STATE gstate = PyGILState_Ensure();
            // py::gil_scoped_acquire acquire;
            float reward ;
            std::vector<float> obs(OBSERVATION_SPACE_DIM, 0.0);
            if (! has_class){
                std::cout<<"Module "<< RUNTIME_MODULE <<" was not found. Operation Reset() not valid."<<std::endl;
                WriteState(reward, obs);
                return;
            }
            try {
                pybind11::gil_scoped_acquire acquire;
                auto ret = env_cls.attr("_reset_envpool")();
                py::array array  = env_cls.attr("_make_obs_envpool")();
                auto r = array.unchecked<float, 1>();  // for 1D arrays — use <2> for 2D, etc
                for (ssize_t i = 0; i < array.size(); ++i) {
                    obs.at(i) = r(i); 
                }
                done =  env_cls.attr("_done_envpool")().cast<bool>();
                reward = env_cls.attr("_reward_envpool")().cast<float>();
            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
            }
            // PyGILState_Release(gstate);
            WriteState(reward, obs);
                    
        }
        bool IsDone() override { 
          return done; 
        }

        void Step(const Action& action) override {

          float reward ;
          std::vector<float> obs(OBSERVATION_SPACE_DIM, 0.0);
          float* action_data = static_cast<float*>(action["action"_].Data());

          if (! has_class){
                std::cout<<"Module "<< RUNTIME_MODULE <<" was not found. Operation Step(...) not valid."<<std::endl;
                WriteState(reward, obs);
                return;
            }
          try {
            
                pybind11::gil_scoped_acquire acquire;
                py::array_t<float> act({ACTION_SPACE_DIM}, action_data); 
                auto ret = env_cls.attr("_step_envpool")(act);
                py::array array  = env_cls.attr("_make_obs_envpool")();
                auto r = array.unchecked<float, 1>();  // for 1D arrays — use <2> for 2D, etc
                for (ssize_t i = 0; i < array.size(); ++i) {
                    obs.at(i) = r(i); 
                }
                done =  env_cls.attr("_done_envpool")().cast<bool>();
                reward = env_cls.attr("_reward_envpool")().cast<float>();
            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
            }
            WriteState(reward, obs);
         }
     private:
        py::module_ mod;
        py::object env_cls;
        bool  done;
        bool has_class;
};


// using WrappedEnvPool = AsyncEnvPool<MODULE_NAME>;

};//namespace wrapper
// using WrappedEnvPool = PyEnvPool<wrapper::WrappedEnvPool>;
// using WrappedEnvSpec = PyEnvSpec<wrapper::WrappedEnvSpec>;
std::string module_name = std::string(XSTR(MODULE_NAME));
std::string spec_name = "_" + module_name +"Spec";
std::string pool_name = "_" + module_name + "Pool";

PYBIND11_MODULE(MODULE_NAME, m) {
  REGISTER(m, PyEnvSpec<EnvSpec<wrapper::SPEC_CLS>>, PyEnvPool<AsyncEnvPool<wrapper::MODULE_NAME>>, spec_name.c_str(), pool_name.c_str())
}
