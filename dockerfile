FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

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
