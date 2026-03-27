MODEL_REGISTRY = {}

def register_model(name):
    def register(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return register