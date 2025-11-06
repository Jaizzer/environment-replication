FROM continuumio/miniconda3:4.5.4

# Create and install dependencies into a Python 3.5 environment
RUN conda create -y -n tfenv python=3.5 && \
    /bin/bash -c "source activate tfenv && \
        pip install --upgrade pip==20.3.4 && \
        # Install the compatible package versions
        pip install \
            tensorflow==1.15.0 \
            keras==2.1.5 \
            numpy==1.16.0 \
            pandas==0.22.0 \
            matplotlib==2.2.2 \
            scikit-learn==0.20.4"

# Use that environment by default
ENV PATH /opt/conda/envs/tfenv/bin:$PATH
ENV CONDA_DEFAULT_ENV=tfenv

# Set working directory
WORKDIR /workspace

# Copy everything in the same directory as this Dockerfile into /workspace
COPY . /workspace

# Default command
CMD ["python"]