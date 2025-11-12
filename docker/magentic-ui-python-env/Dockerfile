FROM continuumio/miniconda3:latest

# Install system dependencies including OpenGL/EGL libraries for ParaView
RUN apt-get update && apt-get install -y \
    ffmpeg \
    exiftool \
    libegl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libgomp1 \
    libsm6 \
    libice6 \
    libxext6 \
    libxrandr2 \
    libxi6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV EXIFTOOL_PATH=/usr/bin/exiftool
ENV FFMPEG_PATH=/usr/bin/ffmpeg
# Force software rendering (same as paraview-docker)
ENV LIBGL_ALWAYS_SOFTWARE=1
# Set display for X11 applications (xvfb will provide virtual display)
ENV DISPLAY=:99

# Accept conda TOS and configure to use only conda-forge channel
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true && \
    conda config --set channel_priority strict && \
    conda config --remove channels defaults || true && \
    conda config --add channels conda-forge

# Install Python 3.10 and ParaView 5.13.3 in base environment
RUN conda install -y -c conda-forge python=3.10 paraview=5.13.3

# Copy requirements and install remaining packages in base environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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