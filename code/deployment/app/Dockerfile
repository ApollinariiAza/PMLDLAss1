FROM python:3.9-slim

WORKDIR /app

# Copy only the Streamlit application files
COPY code/deployment/app /app

# Install dependencies with no cache
RUN pip install --no-cache-dir streamlit requests

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
