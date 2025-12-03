FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    curl \
    xz-utils \
    libfuse2 \
    libusb-1.0-0 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libglu1-mesa \
    libglib2.0-0 \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libegl1-mesa \
    libdbus-1-3 \
    libxss1 \
    libxtst6 \
    libnss3 \
    libasound2 \
    libpulse0 \
    libudev1 \
    libgbm1 \
    libgl1-mesa-glx \
    xvfb \
    mesa-utils \
    python3 \
    python3-pip \
    python3-venv \
    git \
    unzip \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libatk1.0-0 \
    libgdk-pixbuf2.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    alsa-utils \
    pulseaudio

# Download Slippi AppImage
RUN mkdir -p /opt/slippi/ && \
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I_GZz6Xtll2Sgy4QcOQbWK0IcQKdsF5X' -O /opt/slippi/Slippi_EXI_AI.AppImage

# Extract AppImage
RUN cd /opt/slippi && \
    chmod +x ./Slippi_EXI_AI.AppImage && \
    ./Slippi_EXI_AI.AppImage --appimage-extract && \
    mv squashfs-root /opt/slippi-extracted

# Create config directories
RUN mkdir -p /root/.config/SlippiOnline/

# Download and install custom Gecko codes for replay saving
RUN mkdir -p /opt/slippi-extracted/Sys/GameSettings/ && \
    wget -O /opt/slippi-extracted/Sys/GameSettings/GALE01r2.ini \
        https://raw.githubusercontent.com/altf4/slippi-ssbm-asm/libmelee/Output/Netplay/GALE01r2.ini

RUN python3 -m venv /opt/meleeenv && \
    . /opt/meleeenv/bin/activate && \
    python3 -m pip install git+https://github.com/altf4/libmelee.git
    
ENV VIRTUAL_ENV=/opt/meleeenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/echo", "Provide command for headless container."]