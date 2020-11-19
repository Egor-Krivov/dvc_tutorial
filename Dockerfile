FROM mlrepa/wrk_bin_clf_base1:latest

COPY ./requirements.txt /tmp/requirements.txt
RUN sudo pip install --ignore-installed -r /tmp/requirements.txt

RUN sudo jupyter contrib nbextension install && \
    jupyter nbextension enable toc2/main

ARG GIT_CONFIG_USER_NAME
ARG GIT_CONFIG_EMAIL
RUN git config --global user.name $GIT_CONFIG_USER_NAME && \
    git config --global user.email $GIT_CONFIG_EMAIL

WORKDIR /home/binary_clf_device_change
ENV PYTHONPATH=/home/binary_clf_device_change

CMD jupyter-notebook --ip=0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token=''