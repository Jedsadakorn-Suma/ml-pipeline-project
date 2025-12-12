# End-to-End MLOps Pipeline - Fashion MNIST

Production-ready ML API with Training → Tracking → Serving → CI/CD

**Live Demo**: http://127.0.0.1:8000/docs (รัน local)  
**MLflow UI**: http://127.0.0.1:5000  
**Accuracy**: 90.37%

Tech Stack:
- TensorFlow • FastAPI • MLflow • Docker • GitHub Actions

```bash
# Quick Start
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python training/train.py
uvicorn api.main:app --reload
CI/CD
