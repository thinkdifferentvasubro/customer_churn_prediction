"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predictions  # Core ML inference logic


# ================================
# FASTAPI APP
# ================================
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)


# === HEALTH CHECK ===
@app.get("/")
def root():
    return {"status": "ok"}


# ================================
# REQUEST SCHEMA
# ================================
class CustomerData(BaseModel):
    # Demographics
    gender: str
    seniorcitizen: str
    partner: str
    dependents: str

    # Phone services
    phoneservice: str
    multiplelines: str

    # Internet services
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str

    # Account information
    contract: str
    paperlessbilling: str
    paymentmethod: str

    # Numeric features
    tenure: int
    monthlycharges: float
    totalcharges: float


# ================================
# PREDICTION ENDPOINT
# ================================
@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predictions(data.dict())
        prediction = int(result[0])
        return {
            "prediction": "Likely to churn" if prediction == 1 else "Not likely to churn"
        }
    except Exception as e:
        return {"error": str(e)}


# ================================
# GRADIO INTERFACE FUNCTION
# ================================
def gradio_interface(
    gender, seniorcitizen, partner, dependents, phoneservice, multiplelines,
    internetservice, onlinesecurity, onlinebackup, deviceprotection,
    techsupport, streamingtv, streamingmovies, contract,
    paperlessbilling, paymentmethod, tenure, monthlycharges, totalcharges
):
    data = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "tenure": int(tenure),
        "monthlycharges": float(monthlycharges),
        "totalcharges": float(totalcharges),
    }

    result = predictions(data)
    prediction = int(result[0])
    return "Likely to churn" if prediction == 1 else "Not likely to churn"


# ================================
# GRADIO UI
# ================================
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Senior Citizen", value="No"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),

        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No"),

        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="One year"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="No"),
        gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method",
            value="Mailed check"
        ),

        gr.Number(label="Tenure (months)", value=34, minimum=0),
        gr.Number(label="Monthly Charges ($)", value=56.95, minimum=0),
        gr.Number(label="Total Charges ($)", value=1889.5, minimum=0),
    ],
    outputs=gr.Textbox(label="Churn Prediction"),
    title="🔮 Telco Customer Churn Predictor",
    theme=gr.themes.Soft(),
    examples=[
        ["Female", "No", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 85.0, 85.0],
        ["Male", "No", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0]
    ],
)


# ================================
# MOUNT GRADIO INTO FASTAPI
# ================================
app = gr.mount_gradio_app(app, demo, path="/ui")
