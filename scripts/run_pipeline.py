import os
import sys
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Data.load import load_data
from src.Data.preprocess import preprocessor
from src.model.tuning import tune
from src.model.train import training
from src.model.eval import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Churn Training Pipeline")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    tracking_uri = f"file:///{os.path.join(parent_dir, "mlruns").replace(os.sep, '/')}"  # Use forward slashes
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn_logistic_regression")

    with mlflow.start_run():
        # -------- Load data --------
        df = load_data(args.path)

        # -------- Preprocessing --------
        x_train, x_test, y_train, y_test, col_transformer = preprocessor(
            df, args.target
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

main()
