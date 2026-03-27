MODEL_REGISTRY = {}

from protonets.models.registry import MODEL_REGISTRY





def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError("Unknown model: {}".format(name))
    return MODEL_REGISTRY[name](**kwargs)