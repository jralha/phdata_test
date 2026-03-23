FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and training script
COPY app/ ./app/
COPY create_model.py .

# Demographics data needed at inference time; training data is mounted via volume
COPY data/zipcode_demographics.csv ./data/

# model/ is populated either by the train service (volume) or a prior build step
RUN mkdir -p model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]