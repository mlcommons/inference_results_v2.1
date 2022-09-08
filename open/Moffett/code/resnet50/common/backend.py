class BaseBackend:
    def __init__(self, *args, **kwargs):
        pass

    def load_model(self):
        raise NotImplementedError("BaseBackend:load")

    def predict(self):
        raise NotImplementedError("BaseBackend:predict")
