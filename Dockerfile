# SpikeCore Lab — PyTorch → TVM BYOC → Neuromorphic Target
# Multi-stage build: TVM from source + PyTorch CPU + Jupyter

FROM python:3.11-slim AS tvm-builder

# System dependencies for TVM build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    llvm-15 \
    llvm-15-dev \
    libllvm15 \
    clang-15 \
    ninja-build \
    libtinfo-dev \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build TVM
RUN git clone --depth 1 --branch main --recursive \
    https://github.com/apache/tvm /opt/tvm

WORKDIR /opt/tvm/build
RUN cp ../cmake/config.cmake . \
    && echo "set(USE_LLVM /usr/bin/llvm-config-15)" >> config.cmake \
    && echo "set(USE_RELAY_DEBUG ON)" >> config.cmake \
    && cmake -G Ninja .. \
    && ninja -j$(nproc)

# ---------- Runtime stage ----------
FROM python:3.11-slim

# Runtime libs for TVM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    llvm-15-runtime \
    libllvm15 \
    && rm -rf /var/lib/apt/lists/*

# Copy built TVM
COPY --from=tvm-builder /opt/tvm /opt/tvm

# Set TVM environment
ENV TVM_HOME=/opt/tvm
ENV PYTHONPATH=/opt/tvm/python:/lab:${PYTHONPATH}
ENV TVM_LIBRARY_PATH=/opt/tvm/build

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /lab
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=spikecore"]
