import os
import pandas as pd
import mlflow


#loading model
current_dir = os.path.dirname(os.path.abspath(__file__))
model = mlflow.sklearn.load_model(os.path.join(current_dir, "model"))

def predictions(Data):
    try:
        data = pd.DataFrame([Data])
        data.columns = data.columns.str.strip().str.lower()
        return model.predict(data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    

