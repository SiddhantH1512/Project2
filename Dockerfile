# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Flask app and the src directory into the container at /app
COPY ./Bank_functions/Flask-app /app
COPY ./Bank_functions/src /app/src

# Install any needed packages specified in requirements.txt
COPY ./Bank_functions/requirements.txt /requirements.txt
RUN pip install --trusted-host pypi.python.org -r /requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Run app.py when the container launches
CMD ["flask", "run"]
