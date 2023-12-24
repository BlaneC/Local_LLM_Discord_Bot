# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install bash and other useful tools
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run bot.py when the container launches
CMD ["python", "bot.py"]
