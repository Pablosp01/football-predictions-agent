# docker build -t football-predictions-agent .
# docker run -p 8000:8000 -e FOOTBALL_DATA_API_KEY=... football-predictions-agent
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn football_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
