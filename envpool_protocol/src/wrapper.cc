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
#include <thread>
#include <algorithm>
#include "_internal_defs.hh"
#include <stdexcept>
// PYBIND
#include "envpool/core/py_envpool.h"
#include <pybind11/embed.h>
#include "sub_interps.hh" // for multithreading, GIL is annoying, so we need multiple interpreters
#include <numpy/arrayobject.h>

#define MAX_ENVS 100

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



class MODULE_NAME : public Env<EnvSpec<SPEC_CLS>>{
    public:
        MODULE_NAME(const Spec& spec, int envid, std::shared_ptr<sub_interps::sub_interpreter> interp_,
        PyObject *mod): Env<EnvSpec<SPEC_CLS>>(spec, envid) {
          // no python stuff, here or else we will get deadlocks
          this->envid = envid;
          interp = interp_;
          this->mod = mod;

        };

        void WriteState(float reward, const std::vector<float>& obs){
            State state = Allocate();
            for (int i = 0; i < OBSERVATION_SPACE_DIM; ++i) {
              state["obs"_][i] = obs.at(i);

            }
            state["reward"_] = reward;
        }


        void Reset() override { 

            // if (! has_class){
            //   // python stuff goes here               
            //     try {
            //       {
            //         // if (!interp_initializer){
            //         //   // initialize main interpreter
            //         //   // interp_initializer = std::make_unique<sub_interps::initialize>();
            //         //   // thread_scope_enabler = std::make_unique<sub_interps::enable_threads_scope>();
            //         //   // import_array(); 
            //         //   // initialize_static_members();
            //         //   std::cout<<"initialziing inyter[p]"<<std::endl;
            //         //   interp = sub_interps_.at(envid);
            //         //   has_class = true;
            //         // }
            //         // if (!interp){
                      
            //         //   interp = std::make_unique<sub_interps::sub_interpreter>();
            //         // }
            //       }
            // } catch (const std::exception& e) {
            //     std::cout<<e.what()<<std::endl;
            //     has_class = false;
            // }
            // }

            // std::thread t(&wrapper::MyEnv_wrap::Reset_t, this);
            // // maybe detach, but may cause threads to pil up
            // t.join();
            Reset_t();
                    
        }

        private:
        void Reset_t(){
          // check for mod init
          if (! has_class){
              // python stuff goes here
              if (!interp){
                std::cout<<"sub-interp was not properly initialized..This env will do nothing"<<std::endl;
                return;
              }               
                try {
                  {
                    sub_interps::sub_interpreter::thread_scope scope(interp->interp());
                    
                    if (!mod) {
                        PyErr_Print();
                        std::cerr << "Failed to import module " << RUNTIME_MODULE << "\n";
                        return;
                    }

                    // 4) Get your class
                    PyObject* env_cls = PyObject_GetAttrString(mod, PROTOCOL_CLS);
                    if (!env_cls || !PyCallable_Check(env_cls)) {
                        PyErr_Print();
                        std::cerr << "Failed to get class " << PROTOCOL_CLS << "\n";
                        Py_XDECREF(mod);
                        return;
                    }

                    // 5) Build arguments tuple and call the constructor
                    PyObject *args = Py_BuildValue("(i)", envid);
                    this->env  = PyObject_CallObject(env_cls, args);
                    Py_DECREF(args);
                    if (!env) {
                        PyErr_Print();
                        std::cerr << "Failed to instantiate " << PROTOCOL_CLS << "\n";
                        Py_DECREF(env_cls);
                        Py_DECREF(mod);
                        return;
                    }


                    has_class = true;
                  }
            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
                has_class = false;
            }
            
            }
          // runs Reset in async       
          float reward ;
          std::vector<float> obs(OBSERVATION_SPACE_DIM, 0.0);
          if (! has_class){
                std::cout<<"Module "<< RUNTIME_MODULE <<" was not found. Operation Reset() not valid."<<std::endl;
                WriteState(reward, obs);
                return;
            }
            try {
                // sub_interps::sub_interpreter::thread_scope scope(interp->interp());
                sub_interps::sub_interpreter::thread_scope scope(interp->interp());
                PyObject* ret = PyObject_CallMethod(this->env , (char*)"_reset_envpool", nullptr);
                if (!ret) {
                    PyErr_Print();
                    std::cerr << "Failed to call _reset_envpool\n";
                    // handle error…
                }
                Py_DECREF(ret);

                // 4) Call env._make_obs_envpool() → should be a numpy.ndarray
                PyObject* obs_obj = PyObject_CallMethod(this->env , (char*)"_make_obs_envpool", nullptr);
                if (!obs_obj) {
                    PyErr_Print();
                    std::cerr << "Failed to call _make_obs_envpool\n";
                    // handle error…
                }
                if (!PyArray_Check(obs_obj)) {
                  std::cerr << "_make_obs_envpool did not return a numpy.ndarray\n";
              } else {
                  // Cast to PyArrayObject
                  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obs_obj);

                  // Get dimensionality & shape
                  int ndim = PyArray_NDIM(arr);
                  npy_intp* shape = PyArray_DIMS(arr);
                  double* data = static_cast<double*>(PyArray_DATA(arr));

                  // Example: print first 10 elements
                  for (int i = 0; i < std::min<npy_intp>(OBSERVATION_SPACE_DIM, PyArray_SIZE(arr)); ++i) {
                      obs.at(i) =  data[i];
                  }
              }
                // py::array array  = env_cls.attr("_make_obs_envpool")();
                // auto r = array.unchecked<float, 1>();  // for 1D arrays — use <2> for 2D, etc
                // for (ssize_t i = 0; i < array.size(); ++i) {
                //     obs.at(i) = r(i); 
                // }
                // done =  env_cls.attr("_done_envpool")().cast<bool>();
                // reward = env_cls.attr("_reward_envpool")().cast<float>();
                PyObject* done_obj = PyObject_CallMethod(this->env , (char*)"_done_envpool", nullptr);
                if (!done_obj) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to call _done_envpool");
                }
                done = PyObject_IsTrue(done_obj);  // 1 if True, 0 if False
                Py_DECREF(done_obj);

                // 2) Call env._reward_envpool() → Python float
                PyObject* rew_obj = PyObject_CallMethod(this->env , (char*)"_reward_envpool", nullptr);
                if (!rew_obj) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to call _reward_envpool");
                }
                double rew_d = PyFloat_AsDouble(rew_obj);
                if (PyErr_Occurred()) {
                    PyErr_Print();
                    throw std::runtime_error("Return value of _reward_envpool is not a float");
                }
                reward = static_cast<float>(rew_d);
                Py_DECREF(rew_obj);

            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
            }
            WriteState(reward, obs);

        };

        void Step_t(float* action_data){
          // runs Steps in async
          // Do not call this directly
          float reward ;
          std::vector<float> obs(OBSERVATION_SPACE_DIM, 0.0);
          if (! has_class){
                std::cout<<"Module "<< RUNTIME_MODULE <<" was not found. Operation Reset() not valid."<<std::endl;
                WriteState(reward, obs);
                return;
            }
            try {
                // sub_interps::sub_interpreter::thread_scope scope(interp->interp());
                
                sub_interps::sub_interpreter::thread_scope scope(interp->interp());
                npy_intp dims[1] = { ACTION_SPACE_DIM };
                PyObject* arg_arr = PyArray_SimpleNewFromData(
                        /*ndim=*/1,
                        dims,
                        NPY_FLOAT,
                        /*data=*/reinterpret_cast<void*>(action_data)
                    );
                if (!arg_arr) {
                    PyErr_Print();
                    std::cout<<"Could not build NumPy array"<<std::endl;
                  }
                PyObject* ret = PyObject_CallMethod(this->env , (char*)"_step_envpool", (char*)"O", arg_arr);
                Py_DECREF(arg_arr);
                if (!ret) {
                    PyErr_Print();
                    std::cerr << "Failed to call _reset_envpool\n";
                    // handle error…
                }
                Py_DECREF(ret);

                // 4) Call env._make_obs_envpool() → should be a numpy.ndarray
                PyObject* obs_obj = PyObject_CallMethod(this->env , (char*)"_make_obs_envpool", nullptr);
                if (!obs_obj) {
                    PyErr_Print();
                    std::cerr << "Failed to call _make_obs_envpool\n";
                    // handle error…
                }
                if (!PyArray_Check(obs_obj)) {
                  std::cerr << "_make_obs_envpool did not return a numpy.ndarray\n";
              } else {
                  // Cast to PyArrayObject
                  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obs_obj);

                  // Get dimensionality & shape
                  int ndim = PyArray_NDIM(arr);
                  npy_intp* shape = PyArray_DIMS(arr);
                  double* data = static_cast<double*>(PyArray_DATA(arr));

                  // Example: print first 10 elements
                  for (int i = 0; i < std::min<npy_intp>(OBSERVATION_SPACE_DIM, PyArray_SIZE(arr)); ++i) {
                      obs.at(i) =  data[i];
                  }
              }
                // py::array array  = env_cls.attr("_make_obs_envpool")();
                // auto r = array.unchecked<float, 1>();  // for 1D arrays — use <2> for 2D, etc
                // for (ssize_t i = 0; i < array.size(); ++i) {
                //     obs.at(i) = r(i); 
                // }
                // done =  env_cls.attr("_done_envpool")().cast<bool>();
                // reward = env_cls.attr("_reward_envpool")().cast<float>();
                PyObject* done_obj = PyObject_CallMethod(this->env , (char*)"_done_envpool", nullptr);
                if (!done_obj) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to call _done_envpool");
                }
                done = PyObject_IsTrue(done_obj);  // 1 if True, 0 if False
                Py_DECREF(done_obj);

                // 2) Call env._reward_envpool() → Python float
                PyObject* rew_obj = PyObject_CallMethod(this->env , (char*)"_reward_envpool", nullptr);
                if (!rew_obj) {
                    PyErr_Print();
                    throw std::runtime_error("Failed to call _reward_envpool");
                }
                double rew_d = PyFloat_AsDouble(rew_obj);
                if (PyErr_Occurred()) {
                    PyErr_Print();
                    throw std::runtime_error("Return value of _reward_envpool is not a float");
                }
                reward = static_cast<float>(rew_d);
                Py_DECREF(rew_obj);

            } catch (const std::exception& e) {
                std::cout<<e.what()<<std::endl;
            }
            WriteState(reward, obs);

        };

        public:
        bool IsDone() override { 
          return done;  
        }

        void Step(const Action& action) override {
          float* action_data = static_cast<float*>(action["action"_].Data());
          // std::thread t(&wrapper::MyEnv_wrap::Step_t, this);
          //   // maybe detach, but may cause threads to pil up
          // t.join();
          Step_t(action_data);
         }
     private:
        PyObject* mod;
        PyObject * env;
        // to ensure Reset is called for initing
        bool  done = true;
        bool has_class;
        int envid;
        // sub interpreter
        std::shared_ptr<sub_interps::sub_interpreter> interp{nullptr};
        // PyInterpreterState* thread_state;
        std::mutex lock;
};


// using WrappedEnvPool = AsyncEnvPool<MODULE_NAME>;

};//namespace wrapper
// using WrappedEnvPool = PyEnvPool<wrapper::WrappedEnvPool>;
// using WrappedEnvSpec = PyEnvSpec<wrapper::WrappedEnvSpec>;
std::string module_name = std::string(XSTR(MODULE_NAME));
std::string spec_name = "_" + module_name +"Spec";
std::string pool_name = "_" + module_name + "Pool";
// std::unique_ptr<sub_interps::initialize> wrapper::MODULE_NAME::interp_initializer = nullptr;
// std::unique_ptr<sub_interps::enable_threads_scope> wrapper::MODULE_NAME::thread_scope_enabler = nullptr;
// std::vector<std::shared_ptr<sub_interps::sub_interpreter>>  wrapper::MODULE_NAME::sub_interps_ = std::vector<std::shared_ptr<sub_interps::sub_interpreter>>();

PYBIND11_MODULE(MODULE_NAME, m) {
  REGISTER(m, PyEnvSpec<EnvSpec<wrapper::SPEC_CLS>>, PyEnvPool<AsyncEnvPool<wrapper::MODULE_NAME>>, spec_name.c_str(), pool_name.c_str())
}
