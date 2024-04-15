# Use a Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the local app directory to the working directory in the container
COPY ./app /app

# Install any needed dependencies specified in requirements.txt (if any)
RUN pip install -r requirements.txt

# Define the commands to run the Python scripts in parallel and then sequentially
CMD ["bash", "-c", "python /app/python_scripts/3A.py && \
    python /app/python_scripts/3B.py && \
    python /app/python_scripts/3C.py && \
    python /app/python_scripts/3D.py && \
    python /app/python_scripts/3E.py && \
    python /app/python_scripts/3F.py && \
    wait && \
    python /app/python_scripts/3G.py && \
    python /app/python_scripts/4.py && \
    python /app/python_scripts/5.py"]