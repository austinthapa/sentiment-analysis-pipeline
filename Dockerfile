# Light weight pythin image
FROM python:3.10-slim

# Set working directory
WORKDIR /app/

# Copt only requirements.txt first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of code
COPY . /app/

# Expose port
EXPOSE 80

# Run Fast api with Unicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]