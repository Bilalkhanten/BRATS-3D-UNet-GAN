"""Small shared helpers: seeding, JSON logging, config handling."""
from __future__ import annotations
import json, random, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # pragma: no cover
        pass


def jsonable(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=jsonable)


def read_json(path):
    with open(path) as f:
        return json.load(f)


class JsonlLogger:
    """Append-only JSON-lines logger. Every training run writes its own raw log,
    which is what the thesis figures are generated from."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kw) -> None:
        kw["wall_time"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(kw, default=jsonable) + "\n")

    def read(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]
