# Build and run this with --platform=linux/amd64 on Apple Silicon
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- Core packages (X/desktop + VNC + noVNC deps + tools) ---
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
      ca-certificates \
      wget curl git \
      xvfb x11vnc openbox xterm \
      x11-apps \
      xdotool scrot imagemagick \
      net-tools netcat-openbsd \
      dumb-init \
      python3 python3-minimal python3-distutils python-is-python3 \
      xfonts-base dbus-x11 \
      libgl1-mesa-dri libglu1-mesa mesa-utils \
      libx11-6 libx11-xcb1 libxext6 libxrender1 libsm6 libice6 \
      libfontconfig1 libfreetype6 libglib2.0-0 libdbus-1-3 \
      libxkbcommon0 libxkbcommon-x11-0 \
      libxcb1 libxcb-render0 libxcb-shm0 libxcb-randr0 libxcb-shape0 \
      libxcb-xfixes0 libxcb-glx0 \
      libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
      libxcb-xinput0 \
      libxcb-util1 libxcb-cursor0 libxcb-xkb1 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxss1 libnss3 libcups2 libdrm2 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# --- Install noVNC + websockify ---
RUN git clone --branch v1.5.0 https://github.com/novnc/noVNC.git /opt/noVNC && \
    git clone --branch v0.12.0 https://github.com/novnc/websockify /opt/noVNC/utils/websockify && \
    ln -s /opt/noVNC/vnc.html /opt/noVNC/index.html

# --- Create non-root user ---
ENV USERNAME=MCPagent
ENV HOME=/home/${USERNAME}
RUN useradd -m -s /bin/bash -d ${HOME} ${USERNAME}

USER ${USERNAME}
WORKDIR ${HOME}

# --- Install ParaView 6.0.0 (Linux, MPI, Python3.10, amd64) ---
ENV PV_VER=5.13.3
ENV PV_FLAVOR=MPI
ENV PV_PY=Python3.10
ENV PV_PKG=ParaView-${PV_VER}-${PV_FLAVOR}-Linux-${PV_PY}-x86_64.tar.gz
ENV PV_DIR=/opt/ParaView-${PV_VER}-${PV_FLAVOR}-Linux-${PV_PY}-x86_64

RUN mkdir -p ${HOME}/.cache/pv && \
    wget -q -O ${HOME}/.cache/pv/${PV_PKG} https://www.paraview.org/files/v${PV_VER%.*}/${PV_PKG}
USER root
RUN tar -xzf /home/${USERNAME}/.cache/pv/${PV_PKG} -C /opt && \
    ln -s ${PV_DIR} /opt/paraview && \
    ln -s /opt/paraview/bin/paraview  /usr/local/bin/paraview && \
    ln -s /opt/paraview/bin/pvserver  /usr/local/bin/pvserver && \
    ln -s /opt/paraview/bin/pvpython  /usr/local/bin/pvpython && \
    rm -f /home/${USERNAME}/.cache/pv/${PV_PKG}

# --- Config: display + defaults ---
ENV DISPLAY=:1
ARG WIDTH=1280
ARG HEIGHT=800
ENV WIDTH=${WIDTH}
ENV HEIGHT=${HEIGHT}
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_QPA_PLATFORM=xcb
ENV XDG_RUNTIME_DIR=/tmp/xdg
ENV QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb

# --- Bring in the startup script and auto_connect script ---
# (copy as root, keep root:root, just ensure it's executable)
COPY start.sh /usr/local/bin/start.sh
#COPY auto_connect.py /usr/local/bin/auto_connect.py
RUN chmod +x /usr/local/bin/start.sh

# --- Configure ParaView settings to disable welcome dialog and set white background ---
# Create ParaView config directory and copy settings files
RUN mkdir -p ${HOME}/.config/ParaView && \
    chown -R ${USERNAME}:${USERNAME} ${HOME}/.config
# Copy the INI file for Qt settings (window size, welcome dialog, etc.)
COPY --chown=${USERNAME}:${USERNAME} ParaView5.13.3.ini ${HOME}/.config/ParaView/ParaView5.13.3.ini

# --- Networking ---
EXPOSE 6080/tcp 5900/tcp 11111/tcp

# --- Run under a simple init to reap zombies ---
USER ${USERNAME}
ENTRYPOINT ["/usr/bin/dumb-init","--"]
CMD ["/usr/local/bin/start.sh"]