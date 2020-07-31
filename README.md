# Motion Illusions

Support library for 2020 Telluride Neuromorphic Engineering Workshop Challenge "Insights into the Early Motion Pathway (VISION)"

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

