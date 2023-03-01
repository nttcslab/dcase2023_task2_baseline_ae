FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get --yes install libsndfile1

ADD requirements.txt ./
RUN pip install -U pip
RUN pip install -r requirements.txt

ARG USERNAME 
ARG GROUPNAME 
ARG UID
ARG GID
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
USER $USERNAME
