# Slampy
 
## A hierarchical bag of words vision model SLAM pipeline with GPU acceleration. 

![ Slampy Logo](./resources/slampy-logo.png)

# Local installation

- Create venv

`python3 -m venv env`

- Install venv requirements

`pip3 install -r  pip.freeze`

- Install local package

`pip install .`

- Install Jax GPU

```
https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- Install CUDA Toolkit 12.3

```
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
```

- Install cuDNN v8.9.7 

https://developer.nvidia.com/rdp/cudnn-archive

install the correct deb and then...

sudo apt update
sudo apt install libcudnn8
sudo apt install libcudnn8-dev
sudo apt install libcudnn8-samples

help from https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202


# Running tests after update


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:/usr/lib/:/usr/lib64/:/home/jonathan/code/dronium-vision-2/env/lib/


for cuda

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.3/lib64

add this to your bash rc

export PATH=/usr/local/cuda-12.3/bin:$PATH

make sure to start a new terminal or type bash

