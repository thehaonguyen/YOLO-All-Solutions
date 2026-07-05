# Start with Ubuntu 22.04
FROM ubuntu:22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
# Includes python3, pip, git, ffmpeg (needed for video processing), and CUDA libs
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment (optional but recommended)
# We will install packages directly to the system python for simplicity in devcontainer
# but you can create a venv here if you prefer.

# Install Python dependencies
# We copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /workspace

# Default command (Streamlit will be started by VS Code usually, but we set a default)
CMD ["/bin/bash"]