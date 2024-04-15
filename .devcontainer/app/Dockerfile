# Use a base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /usr/src/app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the contents of the local app directory to the working directory in the container
COPY ./app /usr/src/app

# Install any needed dependencies specified in requirements.txt (if any)
RUN pip3 install -r requirements.txt

