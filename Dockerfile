FROM python:3.12.9-slim

WORKDIR /churn_pred_app
COPY app/ /churn_pred_app/app
COPY requirements-docker.txt /churn_pred_app/requirements-docker.txt
COPY src/serving  /churn_pred_app/src/serving

RUN pip install --no-cache-dir -r requirements-docker.txt

EXPOSE 8080
CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8080"]
