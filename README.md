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
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- Install CUDA Toolkit 11.8

```
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
```

# Running tests after update

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:/usr/lib/:/usr/lib64/:/home/jonathan/code/dronium-vision-2/env/lib/