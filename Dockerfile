# Use Python 3.9 with a slim base image to keep the container size small
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from your local machine to the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Inform Docker that the container will listen on port 5000
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_DEBUG=false

# Use an environment variable to determine the command
ARG ENV=production
ENV ENV=${ENV}

# Default to production command
CMD if [ "$ENV" = "development" ]; then flask --app app.py run --host=0.0.0.0; else gunicorn -w 4 -b 0.0.0.0:5000 app:create_app(); fi
