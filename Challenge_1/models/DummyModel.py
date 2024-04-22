class DummyModel:
    def __init__(self):
        pass

    def predict(self, X):
        return [1 for _ in range(X.shape[0])]

    def train(self, X, y):
        pass
