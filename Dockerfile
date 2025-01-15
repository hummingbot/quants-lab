# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Copy the current directory contents and the Conda environment file into the container
COPY core/ core/
COPY environment.yml .
COPY research_notebooks/ research_notebooks/
COPY controllers/ controllers/
COPY tasks/ tasks/

# Create the environment from the environment.yml file
# If cchardet fails, we'll install it separately
RUN conda env create -f environment.yml

# Activate the environment and install cchardet separately if it failed
# RUN conda run -n quants_lab pip install cchardet || echo "cchardet installation failed, continuing anyway"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Copy task configurations
COPY config/tasks.yml /quants-lab/config/tasks.yml

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "run_tasks.py"]