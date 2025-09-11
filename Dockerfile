# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies (this layer will be cached unless system deps change)
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Copy ONLY the environment file first (for dependency caching)
# This layer will only rebuild if environment.yml changes
COPY environment.yml .

# Create the conda environment (this is the expensive step we want to cache)
# This layer will be cached unless environment.yml changes
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Copy the core framework and CLI (these change more frequently)
# This layer will rebuild when code changes, but dependencies remain cached
COPY core/ core/
COPY cli.py .

# Create outputs directory under app/
RUN mkdir -p app/outputs/notebooks

# Optional: Pre-compile Python bytecode for faster startup
RUN python -m compileall core/ || true

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "cli.py", "run", "--config", "config/notebook_tasks.yml"]