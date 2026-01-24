from sklearn.linear_model import LogisticRegression

def training(best_params ,x, y):
    model = LogisticRegression(**best_params)
    model.fit(x, y)

    return model