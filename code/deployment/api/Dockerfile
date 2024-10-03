FROM python:3.9-slim

WORKDIR /app

# Copy the FastAPI application code into the container
COPY code/deployment/api /app

# Create the models directory inside the container
RUN mkdir -p /app/models

# Copy the model from the models folder into the container
COPY code/models/wine_model.pkl /app/models/wine_model.pkl

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn numpy

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
