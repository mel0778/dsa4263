# Use a Python base image
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy all files from the python_scripts directory to the working directory in the container
COPY ./python_scripts /app

# Install any dependencies required for the Python scripts
RUN pip install -r requirements.txt

# Run the Python files in parallel
CMD ["python", "file1.py", "&", "python", "file2.py", "&", "python", "file3.py", "&", "python", "file4.py", "&", "python", "file5.py", "&", "python", "file6.py", "&", "python", "script1.py", "&&", "python", "script2.py", "&&", "python", "script3.py"]