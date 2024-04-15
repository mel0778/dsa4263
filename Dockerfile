# Use a Python base image
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy all files from the python_scripts directory to the working directory in the container
COPY ./python_scripts /app

# Install any dependencies required for the Python scripts
RUN pip install -r requirements.txt

# Run the Python files in parallel then sequentially
CMD ["sh", "-c", "python 3A.py & python 3B.py & python 3C.py & python 3D.py & python 3E.py & python 3F.py & python 3G.py && python 4.py && python 5.py"]
