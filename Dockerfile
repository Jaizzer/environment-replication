FROM continuumio/miniconda3:4.5.4

# --- SECTION 1: SYSTEM FIXES AND COMPILER INSTALLATION (For cffi, pystan, etc.) ---

# 1. Fix Debian Archive Sources (Addressing 404 and expired key errors)
# This MUST be done first because the base OS (Debian 9/Stretch) is End-of-Life (EOL).
# We chain these commands for efficiency and to resolve the source list issues.
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list && \
    sed -i '/security/d' /etc/apt/sources.list

# 2. Update and Install System Dependencies
# We use --allow-unauthenticated because the archive keys are expired.
# This installs GCC/G++ (build-essential) and Boost libraries (for pystan).
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
        libssl-dev \
        libboost-all-dev \
        build-essential && \
    apt-get clean

# --- SECTION 2: PYTHON ENVIRONMENT AND PACKAGE INSTALLATION ---

# Create and install dependencies into a Python 3.5 environment
RUN conda create -y -n tfenv python=3.5 && \
    /bin/bash -c "source activate tfenv && \
        pip install --upgrade pip==20.3.4 && \
        # Install the compatible package versions, including complex ones
        pip install \
            tensorflow==1.15.0 \
            keras==2.1.5 \
            numpy==1.16.0 \
            pandas==0.22.0 \
            matplotlib==2.2.2 \
            scikit-learn==0.20.4 \
            # Install complex dependencies separately to manage build order
            pystan==2.19.1.1 \
            holidays==0.9.12 \
            fbprophet \
            statsmodels"

# --- SECTION 3: ENVIRONMENT AND STARTUP ---

# Use that environment by default
ENV PATH /opt/conda/envs/tfenv/bin:$PATH
ENV CONDA_DEFAULT_ENV=tfenv

# Set working directory
WORKDIR /workspace

# Copy everything in the same directory as this Dockerfile into /workspace
COPY . /workspace

# Default command: Use bash so you land at the shell prompt (tfenv)
# This is generally better than defaulting to python
CMD ["/bin/bash"]