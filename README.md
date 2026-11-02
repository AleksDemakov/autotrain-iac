# MLOps IaC Pipeline

End-to-end MLOps pipeline with **Infrastructure as Code**: model training, MLflow tracking, FastAPI serving, and Kubernetes deployment.

## 🚀 Quick Start

### Train locally
```bash
pip install -r requirements.txt
mlflow server --backend-store-uri sqlite:///mlflow.db
python src/train.py
```

### Deploy to Kubernetes
```bash
docker build -t mlops-api -f Dockerfile.api .
docker build -t mlops-train -f Dockerfile.train .
kubectl apply -f k8s/
```

## 🏗️ Stack

- **ML**: scikit-learn, MLflow
- **API**: FastAPI + Uvicorn
- **IaC**: Docker + Kubernetes
- **Data**: Credit scoring dataset

## 🔗 API

- `GET /health` — model status
- `POST /predict` — make prediction
- `/metrics` — Prometheus
