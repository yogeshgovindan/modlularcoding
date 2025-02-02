# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (the default Streamlit port)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
