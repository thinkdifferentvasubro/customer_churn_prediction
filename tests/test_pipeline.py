import os
import sys
import pandas as pd
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
    path = get_data_path()
    df = load_data(path)
    
    assert df.shape[0] > 0 and df.shape[1] > 0, "data loading failed"
    
        # -------- Preprocessing --------
    x_train, x_test, y_train, y_test, col_transformer = preprocessor(
            df, target
        )
    
    assert set(y_train).issubset({0, 1}) and set(y_test).issubset({0, 1}), "Target contains values other than 0 and 1"
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert not pd.isnull(x_train).any().any(), "NaNs in x_train"
    assert not pd.isnull(x_test).any().any(), "NaNs in x_test"

    

        # -------- Hyperparameter tuning --------
    best_params, best_value = tune(x_train, y_train)

        # log best params

        # -------- Model training --------
    assert best_value > 0.60, "model is not accurate enough" 
    

if __name__=="__main__":
    main()
