FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# เปิด port
EXPOSE 8080

# run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]