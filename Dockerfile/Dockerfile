FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
MAINTAINER Kishwar Shafin, kishwar.shafin@gmail.com

# update and install dependencies
RUN apt-get update && \
    apt-get -y install git cmake make gcc g++ autoconf bzip2 lzma-dev zlib1g-dev \
    apt-get -y install libcurl4-openssl-dev libpthread-stubs0-dev libbz2-dev liblzma-dev libhdf5-dev \
    apt-get -y install python3-pip python3-virtualenv virtualenv && \
    apt-get clean && \
    apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# get HELEN
WORKDIR /opt
RUN git clone https://github.com/kishwarshafin/helen.git && \
    cd /opt/helen && \
    git fetch && \
    git submodule update --init && \
    git pull origin master && \
    make install

WORKDIR /opt/helen

# setup entrypoint
COPY helen_wrapper.sh /opt/helen_wrapper.sh
WORKDIR /data
ENTRYPOINT ["sh", "/opt/helen_wrapper.sh"]