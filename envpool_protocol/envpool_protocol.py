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

"""This module defines the protocol that all envs shall satisfy in order to be wrapped into an envpool
compatible env."""
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class EnvPoolProtocol(ABC):
    """Abstract class, all envs shall inherit from this to be wrapped"""
    
    @abstractmethod
    def _step_envpool(self, action) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def _reward_envpool(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def _reset_envpool(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def _make_obs_envpool(self) -> NDArray:
        raise NotImplementedError() 
    
    @abstractmethod
    def _done_envpool(self) -> bool:
        raise NotImplementedError()
    