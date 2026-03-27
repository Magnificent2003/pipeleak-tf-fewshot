from .datasets import make, collate_fn
from . import mini_imagenet
from . import tiered_imagenet
from . import cifar100
from . import cub200
from . import inatural
from . import transforms
from .custom_dataset import FewShotDataset

def make(name, **kwargs):

    if name == 'custom':
        return FewShotDataset(**kwargs)

    else:
        raise ValueError("Unknown dataset")