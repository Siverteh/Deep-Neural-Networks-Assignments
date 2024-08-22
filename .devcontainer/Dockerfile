# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install necessary tools including Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Expose port 8000 if your project requires it (for example, for a web server)
EXPOSE 8000

# Run a command that keeps the container alive (e.g., a web server)
CMD ["python", "main.py"]  # Replace with your actual start script
