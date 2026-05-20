import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Data.load import load_data
from src.Data.preprocess import preprocessor
from src.model.tuning import tune
from src.model.train import training
from src.model.eval import evaluation


def get_data_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "..", "Data", "churn.csv")
    return path

def main(target="churn"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Customer churn prediction")

    with mlflow.start_run():
        # -------- Load data --------
        data_path = get_data_path()
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data file not found")
        
        df = load_data(data_path)

        # -------- Preprocessing --------
        x_train, x_test, y_train, y_test, col_transformer = preprocessor(
            df, target
        )

        # -------- Hyperparameter tuning --------
        best_params, best_value = tune(x_train, y_train)

        # log best params
        mlflow.log_params(best_params)

        # -------- Model training --------
        model = training(best_params, x_train, y_train)

        # -------- Evaluation --------
        # Make sure evaluation() returns precision, recall, f1, roc in that order
        precision, recall, f1, roc = evaluation(model, x_test, y_test)

        # log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc", roc)

        # -------- Full pipeline --------
        pipeline = Pipeline(steps=[
            ("preprocess", col_transformer),
            ("model", model)
        ])

        # log model pipeline
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print("Training complete.")
        print(f"Evaluation metrics:\nPrecision: {precision}, Recall: {recall}, F1: {f1}, ROC: {roc}")

if __name__=="__main__":
    main()
