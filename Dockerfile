FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TWINDYN_DATA_ROOT=/data

WORKDIR /opt/twindyn

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN pip install --no-cache-dir .

RUN mkdir -p /data /opt/twindyn/runs

CMD ["python", "-m", "twindyn.cli.train", "experiment=main"]
