FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir kaggle-environments

# Override restricted default C++ engine with the custom full-featured cg-lib
COPY remotethings/cg_custom/cg /usr/local/lib/python3.10/site-packages/kaggle_environments/envs/cabt/cg

COPY . /app/

CMD ["python", "test_cabt_docker.py"]
