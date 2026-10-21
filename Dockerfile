# CPU image for tests, examples and the smoke test.  For GPU training use a CUDA base image and a CUDA build of PyTorch.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY requirements.txt pyproject.toml README.md ./
COPY bratsgan ./bratsgan
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt && pip install -e .
COPY . .
CMD ["pytest"]
