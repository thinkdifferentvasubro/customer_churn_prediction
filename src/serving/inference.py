import pickle
import os
import pandas as pd



def predictions(Data):
    """
    Load a model from a pickle file and make predictions on the given data.

    Parameters:
        Data: array-like, input features for prediction
        model_path: str, path to the pickle model file

    Returns:
        predictions: model predictions or None if an error occurs
    """
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model.pkl")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None 

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading model: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None

    try:
        data = pd.DataFrame([Data])
        data.columns = data.columns.str.strip().str.lower()
        return model.predict(data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
