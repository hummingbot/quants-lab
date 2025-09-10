# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Copy only core framework and CLI - other directories should be mounted
COPY core/ core/
COPY environment.yml .
COPY cli.py .

# Create the environment from the environment.yml file
# If cchardet fails, we'll install it separately
RUN conda env create -f environment.yml

# Activate the environment and install cchardet separately if it failed
# RUN conda run -n quants_lab pip install cchardet || echo "cchardet installation failed, continuing anyway"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Create outputs directory
RUN mkdir -p outputs/notebooks

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "cli.py", "run", "--config", "config/notebook_tasks.yml"]