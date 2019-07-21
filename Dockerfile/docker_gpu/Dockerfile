FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
MAINTAINER Kishwar Shafin, kishwar.shafin@gmail.com

# update and install dependencies
RUN apt-get update && \
    apt-get -y install time git make wget autoconf gcc g++ zlib1g-dev libcurl4-openssl-dev libbz2-dev && \
    apt-get -y install python3 python3-dev python3-pip && \
    apt-get clean && \
    apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install cmake
WORKDIR /tmp
RUN mkdir /opt/cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    sh /tmp/cmake-3.14.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# get HELEN
WORKDIR /opt
RUN git clone https://github.com/kishwarshafin/helen.git && \
    cd /opt/helen && \
    git fetch && \
    git submodule update --init && \
    rm -rf /opt/helen/.git

WORKDIR /opt/helen/build
RUN cmake .. -Wno-deprecated && make

RUN python3 -m pip install h5py tqdm numpy pyyaml
# install torch
# RUN python3 -m pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
# RUN python3 -m pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

WORKDIR /opt/helen

# setup entrypoint
COPY helen_wrapper.sh /opt/helen_wrapper.sh
WORKDIR /data
ENTRYPOINT ["sh", "/opt/helen_wrapper.sh"]
