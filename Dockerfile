# Use a small official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy files into the container
COPY kimina-lean-server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the script by default
CMD ["python", "main.py"]