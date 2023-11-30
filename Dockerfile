# Use an official Python runtime as a parent image
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /TP_final_OUNI_HOURANY_TFAILY

# Copy the current directory contents into the container at /app
COPY . /TP_final_OUNI_HOURANY_TFAILY

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run main.py when the container launches
CMD ["python", "./main.py"]
