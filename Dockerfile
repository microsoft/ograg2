FROM msrhubroot.azurecr.io/cliwrapper:latest

WORKDIR /app/ograg2

# Install build tools needed for compiling Python packages with C extensions
RUN tdnf update -y && tdnf install -y \
    build-essential \
    gcc \
    glibc-devel \
    binutils \
    python3-devel \
    && tdnf clean all

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -Ur requirements.txt

# Copy application code
COPY . .

ENV CLI_COMMAND="bash" \
    APP_TITLE="OG RAG Service" \
    CLI_WORKING_DIR="/app/ograg2"
