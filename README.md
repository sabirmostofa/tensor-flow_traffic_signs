# tensor-flow_traffic_signs

Download all dataset Training and Test
run load_data.sh

```
./load_data.sh
```

## Problem 1
images consist of different shape
need to resize all images, so that you can create neural network


## Running on Docker

The following docker notebook image is built over tensorflow notebook https://hub.docker.com/r/jupyter/tensorflow-notebook/

It contains the git repo and downloaded images from load_data.sh

Command to get the docker image:

```
docker pull sabirmostofa/notebook
```

Run the notebook:

```
docker run -it --rm -p 8888:8888 sabirmostofa/notebook jupyter-notebook
```

Run as root:

```
sudo docker run -it --rm --user root -p 8888:8888 sabirmostofa/notebook jupyter-notebook
```

Run bash to update or change git project:

```
sudo docker run -it --rm -p 8888:8888 sabirmostofa/notebook bash
```
