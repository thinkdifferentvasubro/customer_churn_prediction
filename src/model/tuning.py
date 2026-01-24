import optuna
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def tune(X, y):
    """
    Docstring for tune
    
    :param X: Description
    :param y: Description

    returns best param_grid
    """
    #objective function
    def objective(trial, X, y):
        penalty = trial.suggest_categorical(
            "penalty", ["l1", "l2", "elasticnet"]
            )
        param_grid = {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "penalty": penalty,
            "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
            ),
            }
        if penalty == "elasticnet":
            param_grid["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.9)
            param_grid["solver"] = "saga"
        elif penalty == "l1":
            param_grid["solver"] = "saga"
        else:
            param_grid["solver"] = "lbfgs"
        model = LogisticRegression(**param_grid)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="precision_weighted"
            )
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)

    return study.best_params, study.best_value