# FROM python:3.11
# WORKDIR /usr/src/app
# COPY . .
# RUN apt-get update && apt-get install -y git wget nano

# RUN pip3 install --no-cache-dir -r requirements.txt

# CMD ["bash"]

# # docker build -t larry:nnwipqc .
# # docker run -d --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results -v ./pca/results:/usr/src/app/pca/results -v ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation
# # docker run -it --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results ./pca/results:/usr/src/app/pca/results -v ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation
# # docker run -it --rm --gpus "device=3" -v ./results larry:nnwipqc



# # docker run --gpus 1 -it --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results -v ./pca/results:/usr/src/app/pca/results ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation
# # curl "https://drive.usercontent.google.com/download?id={FILE_ID}&confirm=xxx" -o FILENAME.zip


# Use an NVIDIA CUDA base image.
# As discussed, 12.2.2-devel-ubuntu22.04 is a good stable choice for CUDA 12.x
# and supports A100. It's compatible with your detected CUDA 12.6.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive mode for apt and define Python version (your current is 3.11)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# --------------------------------------------------------------------------
# Install System Dependencies and Python Environment
# --------------------------------------------------------------------------
# Ensure python3.11 and venv are installed if they are not the default in the base image.
# python3-pip is often default, but good to be explicit.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3.11-venv \
    curl \
    git \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment for isolated dependencies
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# --------------------------------------------------------------------------
# Install Python Packages from requirements.txt
# (Special handling for torch and qiskit-aer-gpu if needed)
# --------------------------------------------------------------------------

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install PyTorch first, using the specific CUDA wheels, as it's often sensitive.
# Based on your PyTorch CUDA version 12.6 detection, the cu121 wheels are the standard compatible ones.
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Now, install the rest of your requirements.
# We'll remove torch, torchvision, torchaudio, and qiskit-aer-gpu from requirements.txt
# because we're installing them explicitly to control their CUDA dependency.
# qiskit-aer-gpu is installed specifically as the 'cuda12' version.

# Create a temporary requirements file excluding torch/qiskit-aer-gpu to avoid conflicts/reinstalls
RUN grep -v -E '^(torch|qiskit-aer-gpu|torchvision|torchaudio)' requirements.txt > /tmp/filtered_requirements.txt

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Explicitly install qiskit-aer-gpu with its specified version.
# Since the base image is CUDA 12.x, 'qiskit-aer-gpu' (without -cu11) is the correct variant.
RUN pip install --no-cache-dir qiskit-aer-gpu==0.15.1

# --------------------------------------------------------------------------
# Copy your main application code
# --------------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------------
# Set the default command when the container runs
# --------------------------------------------------------------------------
# CMD ["bash"] # Use this if you want to enter the container shell for debugging
# CMD ["python", "your_main_script.py"] # Replace with your actual script name