# Start from a specific Python version for reproducibility
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Update the package list and install git
# -y flag automatically answers "yes" to any prompts
RUN apt-get update && apt-get install -y git

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# The docker-compose.yml file handles mounting the project code into /app,
# so we don't need to COPY it here.

# This command keeps the container running, so the dev container session remains active.
CMD ["tail", "-f", "/dev/null"]