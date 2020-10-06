FROM bitnami/pytorch:1.6.0
# Needs to be set to root if we want to install more dependencies in CI
USER root
RUN echo 'fastai 1.*' >> /opt/bitnami/miniconda/conda-meta/pinned
RUN conda install -y -c fastai -c pytorch fastai torchvision
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
EXPOSE 5001

ADD ./corescore ./corescore
ADD ./tests ./tests

CMD ["gunicorn", "-w", "5", "-b", "0.0.0.0:5001", "corescore.api:app"]
