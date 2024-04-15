# Use a Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the local app directory to the working directory in the container
COPY ./app /app

# Install any needed dependencies specified in requirements.txt (if any)
RUN pip3 install -r requirements.txt

# Define the commands to run the Python scripts in parallel and then sequentially
CMD ["bash", "-c", "python3 -W ignore python_scripts/3A.py &\
    python3 -W ignore python_scripts/3B.py &\
    python3 -W ignore python_scripts/3C.py &\
    python3 -W ignore python_scripts/3D.py &\
    python3 -W ignore python_scripts/3E.py &\
    python3 -W ignore python_scripts/3F.py &\
    wait && \
    python3 -W ignore python_scripts/3G.py"]