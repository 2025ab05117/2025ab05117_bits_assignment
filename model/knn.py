from sklearn.neighbors import KNeighborsClassifier

def train(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model