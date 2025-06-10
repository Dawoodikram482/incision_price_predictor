FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

CMD ["flask", "--app", "app.py", "run", "--host=0.0.0.0", "--port=5050", "--debug"]