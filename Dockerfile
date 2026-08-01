FROM python:3.11-slim

WORKDIR /app

#install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#application code
COPY src/ ./src/

#demo ships with the synthetic model and recall list only
COPY models/synthetic/ ./models/synthetic/
COPY outputs/synthetic/ ./outputs/synthetic/

# Setting DATA_DIR makes FLAVOUR resolve to "synthetic" inside the container,
# so it can only ever load synthetic artifacts.
ENV DATA_DIR=/app/data_working/synthetic
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]