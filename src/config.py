import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path=None):
    with open(path or os.path.join(ROOT, "configs", "default.yaml")) as f:
        return yaml.safe_load(f)


def merged(base, **over):
    c = copy.deepcopy(base)
    c.update(over)
    return c
