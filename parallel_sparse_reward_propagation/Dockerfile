# Use NVIDIA CUDA Base Image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set default command to run a script (e.g., benchmark)
CMD ["python3", "tests/benchmark.py"]
