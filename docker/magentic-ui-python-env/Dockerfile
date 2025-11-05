FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV EXIFTOOL_PATH=/usr/bin/exiftool
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Create environment with Python 3.10 (better ParaView support on ARM64)
RUN conda create -n paraview-env python=3.10 -y

# Activate environment and install ParaView only first
SHELL ["/bin/bash", "-c"]
RUN source activate paraview-env && \
    conda install -c conda-forge paraview -y

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN source activate paraview-env && \
    pip install --no-cache-dir -r requirements.txt

# Set up environment activation for all future commands
ENV PATH=/opt/conda/envs/paraview-env/bin:$PATH
ENV CONDA_DEFAULT_ENV=paraview-env

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
RUN mkdir -p /workspace

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL org.opencontainers.image.source=https://github.com/microsoft/magentic-ui
LABEL org.opencontainers.image.description="Magentic UI Python Environment container for executing code and commands safely"
LABEL org.opencontainers.image.licenses=MIT

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]