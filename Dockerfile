from tensorflow/tensorflow:1.7.1-devel-gpu-py3

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 vim git wget python3-tk

# Would be needed to run pytorch unflow, however the base docker image is mysteriously missing nvrtc shared library files
#RUN pip install torch torchvision cupy-cuda90

COPY requirements.txt /workdir/requirements.txt
WORKDIR /workdir
RUN pip install -r requirements.txt

# Set HOME to workdir which will be mounted from host
# since recommended docker run command uses user uid and gid
# everything in the container is not writeable except this directory
# This is needed so that cupy can store it's cache
ENV HOME=/workdir

# Last step, install motion_illusions in editable mode
# Docker command will mount this directory as volume to provide latest version
COPY . /workdir
RUN pip install -e .
