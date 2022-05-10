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
RUN pip install -r requirements.txt
RUN pip cache purge 