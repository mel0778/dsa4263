# Use a Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the local app directory to the working directory in the container
COPY ./app /app

# Install any needed dependencies specified in requirements.txt (if any)
RUN pip3 install -r requirements.txt

# Define the commands to run the Python scripts in parallel and then sequentially
CMD ["bash", "-c", "cd python_scripts && \
    python3 3A.py & \
    python3 3B.py & \
    python3 3C.py & \
    python3 3D.py & \
    python3 3E.py & \
    python3 3F.py & \
    wait && \
    python3 3G.py &&"]