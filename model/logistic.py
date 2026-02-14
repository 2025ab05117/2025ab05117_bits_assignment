from sklearn.linear_model import LogisticRegression

def train(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model