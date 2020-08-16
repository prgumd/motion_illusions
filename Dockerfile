from nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-dev git python3 python3-pip python3-tk
RUN pip3 install --upgrade pip

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

# We do not actually need to build these CUDA modules, they aren't importable afterwards for some reason
# If you run the container and then build it works well
# RUN cd ./motion_illusions/flownet2-pytorch && bash install.sh
