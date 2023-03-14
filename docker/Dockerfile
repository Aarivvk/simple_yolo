FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 as nvidia


# Install lib torch
RUN apt-get update && apt-get install -y wget unzip && \
    cd /tmp && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip && \
    unzip libtorch-cxx11* && mv libtorch /usr/lib/

RUN apt-get install -y cmake libcudnn8-dev
