"""JSON has no NaN: missing numbers are written as null, so every reader (R's jsonlite too) can parse the file."""
import json

import numpy as np


def clean(x):
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def dumps(x) -> str:
    return json.dumps(clean(x), indent=1, allow_nan=False)
