FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /home/ict/work

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 python3.12-dev \
    default-libmysqlclient-dev build-essential pkg-config graphviz curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

COPY docker_env/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --ignore-installed blinker && \
    pip install -r /tmp/requirements.txt

RUN jupyter notebook --generate-config
COPY docker_env/jupyter_notebook_config.py /root/.jupyter/

EXPOSE 9000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9000", "--no-browser", "--allow-root"]
