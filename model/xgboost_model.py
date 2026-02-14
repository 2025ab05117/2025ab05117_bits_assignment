from xgboost import XGBClassifier

def train(X_train, y_train):
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model