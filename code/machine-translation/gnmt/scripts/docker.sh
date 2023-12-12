#!/usr/bin/env bash

nvidia-docker run -it --rm --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace nvcr.io/nvidia/tensorflow:23.03-tf1-py3
