# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install system libraries needed by OpenCV, then Python packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY . /code/

# Command to run the app when the container launches
# It tells uvicorn to run the 'app' instance from the 'main' file
# It binds to host 0.0.0.0 and port 7860, which is standard for these platforms
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]