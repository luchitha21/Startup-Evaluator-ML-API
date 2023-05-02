FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY mlapi.py /app
COPY requirements.txt /app
COPY catboost.pkl /app

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
EXPOSE 8000
CMD ["uvicorn", "mlapi:app", "--host", "0.0.0.0", "--port", "8000"]