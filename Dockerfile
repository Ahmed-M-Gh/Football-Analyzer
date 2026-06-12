# using lite and efficient Python image
FROM python:3.12-slim

# install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# determine the working directory
WORKDIR /app

# copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose the port the app runs on
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]