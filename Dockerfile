# CRITICAL: ARG RUNTIME_IMAGE is required for MSRHub CLI wrapping.
# The CI pipeline replaces this at build time with the CLI wrapper image.
#
# Local development (standard Python):
#   docker build -t ograg .
#   docker run -it ograg
#
# With CLI wrapper (web terminal):
#   docker build --build-arg RUNTIME_IMAGE=msrhubroot.azurecr.io/cliwrapper:python3.12 -t ograg .
#   docker run -p 8000:8000 ograg
#   Open: http://localhost:8000
ARG RUNTIME_IMAGE=mcr.microsoft.com/azurelinux/base/python:3.12

FROM ${RUNTIME_IMAGE} AS runtime

WORKDIR /app

# Install build tools needed for compiling Python packages with C extensions
RUN tdnf update -y && tdnf install -y \
    build-essential \
    gcc \
    glibc-devel \
    binutils \
    python3-devel \
    && tdnf clean all

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -Ur requirements.txt && \
    pip cache purge 2>/dev/null || true

# Copy application code
COPY . /app/ograg

# CLI wrapper environment variables
ENV CLI_COMMAND="bash" \
    APP_TITLE="OG-RAG: Ontology-Grounded RAG" \
    CLI_WORKING_DIR="/app/ograg"
