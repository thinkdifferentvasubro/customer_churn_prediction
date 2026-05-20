import os
import mlflow
from mlflow.artifacts import download_artifacts


def loading_and_saving(
        tracking_uri="http://127.0.0.1:5000",
        model_uri="runs:/0b3813b06f404e90b2ca739427d91073/model"
        ):
    MLFLOW_TRACKING_URI = tracking_uri

    MODEL_URI = model_uri

    download_path = os.path.abspath(os.path.join("src", "serving"))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Downloading model artifacts from MLflow...")
    model_path = download_artifacts(
        artifact_uri=MODEL_URI,
        dst_path=download_path
    )
    print(f"Model downloaded successfully!")
    print(f"Saved at: {model_path}\n")

if __name__ == "__main__":
    loading_and_saving()