# Motion Illusions

Support library for 2020 Telluride Neuromorphic Engineering Workshop Challenge "Insights into the Early Motion Pathway (VISION)"

A description of some of the illusions of interest is available [here](https://github.com/prgumd/motion_illusions/blob/master/project_overview.pdf).

## Repository Contents

This library implements helper functions for generating data representative of optical illusions humans experience.

The `master` branch containers a Python package called `motion_illusions`. The package contains helper functions for simulating the illusory flow and bias estimation illusions.

The [stepping_feet_illusion_matlab](https://github.com/prgumd/motion_illusions/tree/steppingfeet_illusion_matlab) branch contains Matlab code for generating events from the stepping feet illusion.

To checkout stepping feet after cloning this repo run:
```bash
git fetch origin steppingfeet_illusion_matlab
git checkout steppingfeet_illusion_matlab
```

The [variational-leviant](https://github.com/prgumd/motion_illusions/tree/variational-leviant) branch contains Matlab code for generating illusory patterns inspired by the Leviant illusion.

To checkout stepping feet after cloning this repo run:
```bash
git fetch origin variational-leviant
git checkout variational-leviant
```

## Overview of motion_illusions python package

The library implements image warping, visualization utilities, and wraps the [UnFlow](https://github.com/simonmeister/UnFlow) optical flow estimator.

The image warping is meant to simulate the small image pertubations due to [saccades](https://en.wikipedia.org/wiki/Saccade).

UnFlow is an unsupervised deep optical flow estimator. Currently the library supports using it to evaluate illusory patterns using pre-generated data. The intent is to extend the network to accept multiple frames as would be necessary for enforcing a causality requirement.

### Examples
* `./examples`
  * `rotation_warp_image.py` Demonstrate continuously warp an image by a rotation and plotting utilities
  * `translation_warp_image.py` Demonstrate continuously warp an image by a translation and plotting utilities

### Library files
* `./motion_illusions`
  * `rotation_translation_image_warp.py` Utilities for warping an image based on rotation and translation
* `./motion_illusions/utils`
  * `flow_plot.py` Convert optical flow into an image
  * `image_tile.py` Easily compose multiple images into a single image for display, useful for debugging
  * `rate_limit.py` Rate limit a process by wall clock time (useful for visualization)
  * `signal_plot.py` Produce an image plotting multiple real time signals
  * `time_iterator.py` Iterate over time using a wall clock or fixed timestep

## Dependencies

* Python >= 3.5
* CUDA enabled GPU
* See `requirements.txt` for python dependencies

## Setup

Since training neural networks is the goal it assumed a GPU is available.

Clone the repository, initiate submodules.
```bash
git clone https://github.com/prgumd/motion_illusions.git
cd motion_illusions
git submodule update --init
```

Download FlowNet2-SD pretrained [model weights](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view) and put in `motion_illusions/flownet2-pytorch/checkpoints`. You will need to create the folder.

### Docker
A Dockerfile is provided to build an image with all packages installed.

Some of the code assumes it can display GUIs using an X server. A `docker run` command is provided for Ubuntu host systems that configures the container for X forwarding.

Make sure docker is installed with access to the GPU.
* [Docker](https://docs.docker.com/get-docker/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Run the following to build the docker image.

```bash
cd motion_illusions
docker build --tag motion_illusions:3.0 .
```

To launch the container with X forwarding, GPUs available, and the latest version of this package:
```bash
cd motion_illusions
docker run  -u $(id -u):$(id -g) -e DISPLAY -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" --ipc host --gpus all --rm -it -v $(pwd):/workdir motion_illusions:3.0 bash
```

The container will complain about having no user due to overriding the uid and gid for X forwarding. This is ok for our purposes, it is the simplest way to handle X forwarding without a security compromise. More details available [here](http://wiki.ros.org/docker/Tutorials/GUI).

On first run the FlowNet2 CUDA modules need to built and installed manually. For some reason this is not working from the Dockerfile yet

```bash
cd motion_illusions/flownet2-pytorch
bash install.sh
```

### Virtualenv

Make sure the following is installed on the host.
* CUDA 9.2
* Python >= 3.5
* virtualenv ('pip3 install virtualenv')

Run the following steps to create a virtual environment, activate it, and install packages.

```bash
cd motion_illusions
virtualenv venv
source venv/bin/activate
pip install -r requirements.txtc
pip install -e .
```

## Running Flownet2-SD

### Inference
#### FlowNet2-SD - Flying Chairs SD
Assumes Flying Chairs Small Displacement is extracted to `motion_illusions/flownet2-pytorch/ChairsSDHom`

For some reason EPE is very high with pre-trained weights, something may be wrong with the implementation, more investigation is needed.
Full FlowNet2 model has low error as expected.

```bash
python3 main.py --inference --model FlowNet2SD --save_flow --inference_dataset ChairsSDHomTrain --inference_dataset_root ChairsSDHom/data --resume checkpoints/FlowNet2-SD_checkpoint.pth.tar
```

#### FlowNet2 - Sintel Clean
Assumes Sintel full is downloaded to `motion_illusions/flownet2-pytorch/mpi-sintel`

```bash
python3 main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean --inference_dataset_root mpi-sintel/training --resume checkpoints/FlowNet2_checkpoint.pth.tar
```

### Training
#### FlowNet2-SD - ChairsSDHom - MultiScale

```bash
python3 main.py --batch_size 8 --model FlowNet2SD --optimizer=Adam --loss=MultiScale --loss_norm=L1 --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --training_dataset ChairsSDHomTrain --training_dataset_root ChairsSDHom/data  --validation_dataset ChairsSDHomTest --validation_dataset_root ChairsSDHom/data
```

#### FlowNet2 - Sintel - L1
Assumes Sintel full is downloaded to `motion_illusions/flownet2-pytorch/mpi-sintel`

```bash
python3 main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset MpiSintelFinal --training_dataset_root mpi-sintel/training   --validation_dataset MpiSintelClean --validation_dataset_root mpi-sintel/training
```

#### FlowNet2C - Sintel - MultiScale

```bash
python3 main.py --batch_size 8 --model FlowNet2C --optimizer=Adam --loss=MultiScale --loss_norm=L1 --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --crop_size 384 512 --training_dataset MpiSintelFinal --training_dataset_root mpi-sintel/training  --validation_dataset MpiSintelClean --validation_dataset_root mpi-sintel/training
```

## Contributors

Developed by Cornelia Fermuller, Chethan Parameshwara, and Levi Burner with the [Perception and Robotics Group](http://prg.cs.umd.edu/) at the University of Maryland.

