#FROM nvidia/cuda:11.0-devel-ubuntu18.04
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -qy --upgrade \
	cmake \
	python3 python3-pip python3-setuptools \
	wget \
	libsm6 libxext6 libxrender-dev \
	xauth libgl1-mesa-glx \
	vim


RUN pip3 install --upgrade pip 

RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install pytorch-lightning

RUN pip3 install opencv-python

RUN pip3 install matplotlib

RUN pip3 install scikit-learn

RUN pip3 install albumentations

RUN pip3 install pycocotools

RUN mkdir /src

WORKDIR /src/


