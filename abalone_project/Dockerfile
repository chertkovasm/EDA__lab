FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/trained_model

CMD ["sh", "-c", "python model.py && python app.py & sleep 10 && python generate_dashboard.py && python run_dashboard.py"]