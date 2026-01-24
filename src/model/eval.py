from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluation(model , x_test, y_test):
    preds = model.predict(x_test)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)
    return precision, recall, f1, roc

