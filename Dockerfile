FROM python:3.12-slim

WORKDIR /app

# Create the package directory
RUN mkdir -p /app/travel_pro

# Install dependencies first for caching
COPY server/requirements.txt /app/travel_pro/server/requirements.txt
RUN pip install --no-cache-dir -r /app/travel_pro/server/requirements.txt

# Copy the entire project into the package directory
COPY . /app/travel_pro/

# Set PYTHONPATH to the parent of the package
ENV PYTHONPATH=/app

# Start the environment server
CMD ["uvicorn", "travel_pro.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
