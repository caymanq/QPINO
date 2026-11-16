import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_RECORD = {}


@pytest.fixture
def record(request):
    def _r(**kw):
        _RECORD.setdefault(request.node.name, {}).update({k: float(v) if hasattr(v, "__float__") else v
                                                          for k, v in kw.items()})
    return _r


def pytest_sessionfinish(session, exitstatus):
    if _RECORD:
        path = os.path.join(ROOT, "results", "metrics", "verification_numbers.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(_RECORD, f, indent=2)
