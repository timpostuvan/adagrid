FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y python3.8-dev python3-pip python3-setuptools
RUN apt-get clean
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python

WORKDIR /adagrid
COPY . /adagrid

RUN python -m pip install --upgrade pip
RUN pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-geometric
RUN pip install -r requirements.txt
RUN pip cache purge 