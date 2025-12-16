# Dockerfile

# 1. Use a lightweight Python base image
# Using Python 3.12
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency files and install
# This helps cache the install layer if only .py files change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
# Copy the Detection folder AND the web-app folder
COPY Detection ./Detection
COPY web-app ./web-app

# 5. Expose the port where Flask runs
EXPOSE 5000

# 6. Set Flask environment variables (optional but good practice)
ENV FLASK_APP web-app/app.py
ENV FLASK_ENV production # Use 'development' for auto-reload (less recommended in Docker)

# 7. Command to start the Flask application

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]