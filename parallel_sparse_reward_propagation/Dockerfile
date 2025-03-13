# Use a dynamic base image
ARG BASE_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
FROM ${BASE_IMAGE}

# Define build arguments
ARG PYTHON_VERSION="3.10"
ARG TORCH="torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"

# Use bash shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
WORKDIR /app

# Update and install dependencies
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    git wget curl bash libgl1 software-properties-common && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python if specified
RUN if [ -n "${PYTHON_VERSION}" ]; then \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt install "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" -y --no-install-recommends; \
    fi

# Install pip and dependencies
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir ${TORCH} triton pandas matplotlib jupyterlab notebook

# Set up workspace
COPY . /app

# Set default command to run benchmark
CMD ["python3", "tests/benchmark.py"]
