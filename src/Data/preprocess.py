import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import OutlierTrimmer
from sklearn.model_selection import train_test_split

def preprocessor(df: pd.DataFrame, target="churn"):
    """
    Preprocess dataframe:
    - cleans column names
    - handles missing values
    - caps outliers
    - encodes categoricals
    - scales numerics
    - splits train/test
    """

    # clean column names
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # drop id column if exists
    if "customerid" in df.columns:
        df = df.drop(["customerid"], axis=1)

    # binary target mapping
    df[target] = df[target].replace({"Yes": 1, "No": 0})

    # fix seniorcitizen
    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].replace({0: "No", 1: "Yes"})

    # split X and y
    y = df[target]
    x = df.drop([target], axis=1)

    # fix numeric string column
    if "totalcharges" in x.columns:
        x["totalcharges"] = pd.to_numeric(x["totalcharges"], errors="coerce")

    # detect categorical columns
    cat_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()

    num_cols = []
    cat_with_num_values = []

    for col in x.columns:
        if col not in cat_cols and x[col].nunique() > 15:
            num_cols.append(col)
        elif col not in cat_cols and x[col].nunique() <= 15:
            cat_with_num_values.append(col)

    # pipelines
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_capping", OutlierTrimmer(capping_method="iqr", tail="both", fold=1.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    cat_with_num_values_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # build transformers safely
    transformers = []

    if len(num_cols) > 0:
        transformers.append(("num", num_pipeline, num_cols))

    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if len(cat_with_num_values) > 0:
        transformers.append(("cat_with_num_vals", cat_with_num_values_pipeline, cat_with_num_values))

    preprocessor = ColumnTransformer(transformers=transformers)

    # split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # fit + transform
    preprocessor.fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_train, x_test, y_train, y_test, preprocessor
