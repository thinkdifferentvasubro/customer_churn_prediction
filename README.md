# Customer Churn Prediction System

An end-to-end Machine Learning project focused on predicting customer churn in the telecom domain. This project covers the complete ML lifecycle — from data preprocessing and feature engineering to model tuning, experiment tracking, deployment, containerization, testing, and CI/CD automation.

The goal of this project was not only to build an accurate churn prediction model, but also to design a production-ready machine learning workflow following real-world MLOps practices.

---

# Project Journey

The project started with exploring and preprocessing telecom customer data to identify patterns related to customer churn.

Multiple machine learning models were experimented with, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

After evaluating all models, Logistic Regression performed the best for this dataset. Since the dataset was relatively small, the simpler linear model generalized better and produced more stable results compared to more complex ensemble models, which showed signs of overfitting.

To further improve performance, Optuna was used for automated hyperparameter tuning. All experiments, model parameters, metrics, and trained models were logged using MLflow for reproducibility and experiment tracking.

The final trained model was deployed using FastAPI for backend model serving and integrated with Gradio to provide an interactive web interface for users. The application was containerized using Docker for portability and deployment consistency.

A CI/CD pipeline using GitHub Actions was also implemented to automatically:

- Run validation/testing scripts
- Build Docker images
- Push Docker images to Docker Hub automatically after successful execution

This project demonstrates practical implementation of:

- Machine Learning pipelines
- Hyperparameter optimization
- Experiment tracking
- API development
- Dockerization
- CI/CD automation
- Automated testing
- Production-ready deployment

---

# Features

- End-to-end ML pipeline
- Telecom customer churn prediction
- Data loading and validation
- Robust preprocessing pipeline
- Hyperparameter tuning using Optuna
- Experiment tracking with MLflow
- Logistic Regression model training
- REST API using FastAPI
- Interactive UI using Gradio
- Dockerized deployment
- CI/CD pipeline using GitHub Actions
- Automated Docker image build and push to Docker Hub
- Automated testing and validation
- Modular and production-oriented code structure

---

# Tech Stack

- Python
- Scikit-learn
- Pandas
- Optuna
- MLflow
- FastAPI
- Gradio
- Docker
- GitHub Actions
- feature-engine

---

# Project Structure


---

# Model Training Workflow

1. Load telecom churn dataset  
2. Validate data integrity  
3. Preprocess and transform features  
4. Split training and testing data  
5. Perform hyperparameter tuning using Optuna  
6. Train Logistic Regression model  
7. Evaluate model performance  
8. Log parameters and model artifacts using MLflow  
9. Serve predictions through FastAPI  
10. Build user interface using Gradio  
11. Containerize application using Docker  
12. Automate testing and Docker deployment using CI/CD pipeline  

---

# Testing & Validation

A dedicated testing script was implemented to ensure reliability and robustness of the ML pipeline.

The script validates:

- Successful dataset loading  
- Correct binary target labels  
- Shape consistency between features and labels  
- Absence of missing values after preprocessing  
- Minimum acceptable model accuracy threshold  

This helps prevent failures during deployment and ensures consistent model quality.

---

# CI/CD Pipeline

A CI/CD pipeline was implemented using GitHub Actions to automate the workflow.

Pipeline functionality includes:

- Running validation/testing scripts automatically
- Building Docker images after successful tests
- Pushing Docker images directly to Docker Hub
- Ensuring deployment consistency and reproducibility

This automation reduces manual deployment effort and improves reliability.

---

# Run the Application

## Start FastAPI Server

```bash
uvicorn app:app --reload
```

## Run with Docker

```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

---

# Future Improvements

- Deploy on cloud platforms (AWS/GCP/Azure)
- Add model monitoring and drift detection
- Improve scalability for larger datasets
- Add database integration for prediction logging
- Add authentication and security layers
- Implement advanced MLOps monitoring tools

---

# Author

**Vasu**
