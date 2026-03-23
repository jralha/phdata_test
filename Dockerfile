FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and training script
COPY app/ ./app/
COPY create_model.py .
COPY tests/ ./tests/

# Runtime and test data bundled into the image
COPY data/zipcode_demographics.csv ./data/
COPY data/future_unseen_examples.csv ./data/

# model/ is populated either by the train service (volume) or a prior build step
RUN mkdir -p model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]