# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for h5py and other Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Flask will run on
EXPOSE 8080

# Define environment variable to tell Flask to listen on all interfaces
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application
CMD ["python", "main.py"]
