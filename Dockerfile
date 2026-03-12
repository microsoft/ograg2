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

# ---------- builder stage: compile wheels with build tools ----------
ARG RUNTIME_IMAGE=mcr.microsoft.com/azurelinux/base/python:3.12

FROM ${RUNTIME_IMAGE} AS builder

WORKDIR /build

RUN tdnf update -y && tdnf install -y \
    build-essential \
    gcc \
    glibc-devel \
    binutils \
    python3-devel \
    ca-certificates \
    rust \
    cargo \
    && tdnf clean all

# Create a virtual env so all packages land in one portable directory
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install CPU-only PyTorch first, then the rest of the requirements
# Use uv for fast dependency resolution (graphrag has deep dependency trees)
RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir --only-binary numpy,scipy,pandas -r requirements.txt && \
    uv pip install --no-cache-dir --no-deps azureml-rag && \
    uv pip install --no-cache-dir cloudpickle azure-ai-ml && \
    (pip cache purge 2>/dev/null || true)

# ---------- runtime stage: lean final image ----------
FROM ${RUNTIME_IMAGE} AS runtime

WORKDIR /app

# Copy the virtual env from the builder (includes all packages + scripts)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . /app/ograg

# CLI wrapper environment variables
ENV CLI_COMMAND="bash" \
    APP_TITLE="OG-RAG: Ontology-Grounded RAG" \
    CLI_WORKING_DIR="/app/ograg"
