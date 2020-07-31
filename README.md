# Motion Illusions

Support library for 2020 Telluride Neuromorphic Engineering Workshop Challenge "Insights into the Early Motion Pathway (VISION)"

This library implements helper functions for generating data representative of optical illusions humans experience.

The `master` branch containers a Python package called `motion_illusions`. The package contains helper functions for simulating the illusory flow and bias estimation illusions.

The [stepping_feet_illusion_matlab](https://github.com/prgumd/motion_illusions/tree/steppingfeet_illusion_matlab) branch contains Matlab code for generating events from the stepping feet illusion.

To checkout stepping feet after cloning this repo run:
```bash
git fetch origin steppingfeet_illusion_matlab
git checkout steppingfeet_illusion_matlab`
```

## Overview of motion_illusions python package

The library implements image warping, visualization utilities, and wraps the [UnFlow](https://github.com/simonmeister/UnFlow) optical flow estimator.

The image warping is meant to simulate the small image pertubations due to [saccades](https://en.wikipedia.org/wiki/Saccade).

UnFlow is an unsupervised deep optical flow estimator. Currently the library supports using it to evaluate illusory patterns using pre-generated data. The intent is to extend the network to accept multiple frames as would be necessary for enforcing a causality requirement.

### Examples
* `./examples`
  * `rotation_warp_image.py` Demonstrate continuously warp an image by a rotation and plotting utilities
  * `translation_warp_image.py` Demonstrate continuously warp an image by a translation and plotting utilities
  * `test_unflow.py` Demonstrate the wrapping around UnFlow making it possible to evaluate on custom datasets

### Library files
* `./motion_illusions`
  * `evaluate_unflow.py` Run UnFlow on batch of images and return results
  * `generic_unflow_input.py` A data loader object for interfacing with the UnFlow library
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

Download pretrained [model weights](https://drive.google.com/file/d/16rOMerQvUnj6UjGjMyQayC1GcqaRu44b/view) from UnFlow authors. Put in root of project.

```bash
mkdir -p unflow_logs/ex
unzip -d unflow_logs/ex unflow_models.zip
```

### Docker
A Dockerfile is provided to build an image with all packages installed. This is probably the easiest method to setup due to UnFlow needing an old version of CUDA and tensorflow.

Some of the code assumes it can display GUIs using an X server. A `docker run` command is provided for Ubuntu host systems that configures the container for X forwarding.

Make sure docker is installed with access to the GPU.
* [Docker](https://docs.docker.com/get-docker/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Run the following to build the docker image.

```bash
cd motion_illusions
docker build --tag motion_illusions:1.0 .
docker run --gpus all --rm -it motion_illusions:1.0 -v $(pwd):/workspace bash
```

To launch the container with X forwarding, GPUs available, and the latest version of this package:
```bash
cd motion_illusions
docker run  -u $(id -u):$(id -g) -e DISPLAY -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" --ipc host --gpus all --rm -it -v $(pwd):/workdir motion_illusions:1.0 bash
```

The container will complain about having no user due to overriding the uid and gid for X forwarding. This is ok for our purposes, it is the simplest way to handle X forwarding without a security compromise. More details available [here](http://wiki.ros.org/docker/Tutorials/GUI).

### Virtualenv

Make sure the following is installed on the host.
* CUDA 9.0
* Python >= 3.5
* virtualenv ('pip3 install virtualenv')

Run the following steps to create a virtual environment, activate it, and install packages.

```bash
cd motion_illusions
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Contributors

Developed by Cornelia Fermuller, Chethan Parameshwara, and Levi Burner with the [Perception and Robotics Group](http://prg.cs.umd.edu/) at the University of Maryland.

