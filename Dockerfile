# syntax=docker/dockerfile:1.7-labs
# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Create the environment from the environment.yml file (do first to avoid invalidating the environment layer cache)
COPY environment.yml .
# If cchardet fails, we'll install it separately
RUN conda env create -f environment.yml

# Activate the environment and install cchardet separately if it failed
# RUN conda run -n quants_lab pip install cchardet || echo "cchardet installation failed, continuing anyway"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Copy Optional wheels directory and handle wheel installation (for utilizing local hummingbot version)
COPY --parents wheels* . 

# Optionally install local Hummingbot wheel if present - use the latest wheel only
RUN if [ -n "$(find wheels/ -name 'hummingbot-*.whl' 2>/dev/null)" ]; then \
    echo "Installing local Hummingbot wheel..." && \
    LATEST_WHEEL=$(find wheels/ -name 'hummingbot-*.whl' | sort -r | head -n1) && \
    echo "Using wheel: $LATEST_WHEEL" && \
    pip install --force-reinstall $LATEST_WHEEL && \
    echo "Local Hummingbot wheel installed successfully"; \
    else \
    echo "No local Hummingbot wheel found, using version from environment.yml"; \
    fi

# Can Comment if only running locally since the local volume is mounted in compose file, 
# only really needed for remote deployment 
COPY config/*.yml config/
COPY core/ core/
COPY --parents research_notebooks/*.py research_notebooks/
COPY controllers/ controllers/
COPY tasks/ tasks/
COPY conf/ conf/

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "run_tasks.py"]