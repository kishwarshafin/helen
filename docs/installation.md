# Install MarginPolish and HELEN
`MarginPolish` and `HELEN` can be used in Linux-based system. Users can install `MarginPolish` and `HELEN` on <b>`Ubuntu 18.04`</b> by following this document.

<center>
<h2> Install MarginPolish </h2>
</center>

To install MarginPolish in a Ubuntu/Linux bases system, follow these instructions:

##### Install Dependencies
```bash
sudo apt-get -y install cmake make gcc g++ autoconf bzip2 lzma-dev zlib1g-dev \
 libcurl4-openssl-dev libpthread-stubs0-dev libbz2-dev liblzma-dev libhdf5-dev
```

##### Install marginPolish
```bash
git clone https://github.com/UCSC-nanopore-cgl/marginPolish.git
cd marginPolish
git submodule update --init
```
Make a build directory:
```bash
mkdir build
cd build
```

Generate makefile:
```bash
cmake ..
make
./marginPolish
# to create a symlink
ln -s marginPolish /usr/local/bin
```

<center>
<h2> Install HELEN </h2>
</center>

Although `HELEN` can be used in a `CPU` only machine, we highly recommend using a machine with `GPU`.

This requires installing `CUDA` and the right `PyTorch` version compiled against the installed version of `CUDA`.

##### Install CUDA
To download `CUDA` for `Ubuntu 18.04` follow these insructions:
```bash
# download CUDA for ubuntu 18.04 x86_64 by running:
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
# if you are using other systems please download the correct version from here:
# https://developer.nvidia.com/cuda-10.0-download-archive

# install CUDA by running:
sudo sh cuda_10.0.130_410.48_linux.run
# 1) Read or scroll through the EULA (you can press 'z' to scroll down).
# 2) Accept the EULA, put yes for OpenGL library, CUDA toolkit.
#    Installing CUDA-samples is optional

# once installed, verify the right version by running:
cat /usr/local/cuda/version.txt
# Expected output: CUDA Version 10.0.130

# Verify that you can see your GPU status by running:
nvidia-smi
```

##### Install PyTorch
Please follow the instructions from this [pytorch-installation-guide](https://pytorch.org/get-started/locally/) to install the right `PyTorch` version.

```bash
# if you are using Ubuntu 18.04, python3 version 3.6.7 and CUDA 10.0 then follow these commands:
python3 -m pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
python3 -m pip install torchvision

# otherwise install the right version by following the instructions from:
# https://pytorch.org/get-started/locally/
```

We have tested these PyTorch versions against `HELEN` to ensure GPU accelerated inference:
* PyTorch 1.0 with CUDA 10.0
* PyTorch 1.1 with CUDA 10.0

To ensure `PyTorch` is using `CUDA`, you can follow these instructions:
```bash
$ python3
>>> import torch
>>> torch.cuda.is_available()
TRUE
# the expected output is TRUE
```

#### Install HELEN
`HELEN` requires `cmake` and `python3` to be installed in the system.
```bash
sudo apt-get -y install cmake
sudo apt-get -y install python3
sudo apt-get -y install python3-dev
```
To install `HELEN`:

```bash
git clone https://github.com/kishwarshafin/helen.git
cd helen
./build.sh
```

These steps will install `HELEN` in your local system. `HELEN` also requires installing some python3 packages.
```bash
python3 -m pip install h5py tqdm numpy torchnet pyyaml
```

#### Usage
If you have installed sucessfully then please follow the [Local Install Usage Guide](docs/usage_local_install.md).
.
