ARG DOCKER_ENV=latest

FROM tensorflow/tensorflow:${DOCKER_ENV}
# DOCKER_ENV are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG DOCKER_ENV

ADD . /develop

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git nano graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install models, scripts and utils
RUN pip install --upgrade pip && \
    pip3 install -e /develop && \
    pip3 install -U tensorflow && \
    pip3 install gdown==3.10.0 && \
    pip3 install tensorflow_datasets tensorflowjs && \
    pip3 install seaborn eli5 shap pydot pdpbox sklearn opencv-python IPython prettytable py7zr

WORKDIR /develop
