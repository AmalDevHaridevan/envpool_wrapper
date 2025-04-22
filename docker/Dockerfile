FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
WORKDIR $HOME
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update \
    && apt-get install -y git curl wget gcc-9 g++-9 build-essential make unzip
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN apt-get install python3-openssl libbz2-dev zlib1g-dev python3-dev python3-pip python3-venv zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev libffi-dev liblzma-dev \
llvm xz-utils tk-dev libxml2-dev libxmlsec1-dev qtdeclarative5-dev -y 
ENV MAKEFLAGS="-j10"
RUN curl https://pyenv.run | bash
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile
RUN echo 'eval "$(pyenv init -)"' >> /etc/profile
RUN command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH
RUN  pyenv install -v 3.11-dev
RUN pyenv global 3.11-dev
RUN python -m pip install --upgrade numpy==1.26  gym dm_env pytree optree gymnasium packaging tqdm
RUN apt-get install libeigen3-dev cmake python3-pybind11 -y
RUN git clone https://github.com/google/glog.git /tmp/glog
RUN cmake -S /tmp/glog -B /tmp/build -G "Unix Makefiles"
RUN cmake --build /tmp/build --target install --parallel 10
RUN git clone https://github.com/AmalDevHaridevan/envpool_wrapper.git 
WORKDIR /root/envpool_wrapper
RUN python3.11 example_env.py
RUN python3.11 benchmark.py --n_envs 100 --async --max_iters 1000 > /tmp/results.txt
RUN tail -n10 /tmp/results.txt