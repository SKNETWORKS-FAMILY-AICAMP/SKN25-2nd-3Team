FROM python:3.12.8-slim-bullseye

WORKDIR /home/ict/work

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    python3-dev default-libmysqlclient-dev build-essential pkg-config graphviz

COPY docker_env/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

RUN jupyter notebook --generate-config
COPY docker_env/jupyter_notebook_config.py /root/.jupyter/

EXPOSE 9000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9000", "--no-browser", "--allow-root"]
