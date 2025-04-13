import hashlib
import json
import os
from pathlib import Path

import numpy
import numpy as np
from numpy.typing import ArrayLike

CACHE_DIR = Path(f"{Path(__file__).parent.resolve()}/.cache")
LOCK_FILE = CACHE_DIR / "lock.json"


class Cache(object):
    _instance = None
    _lock: dict

    def __new__(cls):
        if cls._instance is None:
            print('Creating cache')
            cls._instance = super(Cache, cls).__new__(cls)
            if Path(CACHE_DIR).exists():
                with open(LOCK_FILE) as f:
                    cls._lock = json.load(f)
            else:
                os.makedirs(CACHE_DIR)
                cls._lock = {}
                with open(LOCK_FILE, 'w') as f:
                    f.write(json.dumps(cls._lock))
        return cls._instance

    def exists(self, key: str) -> bool:
        return key in self._lock

    def load_np(self, key: str) -> ArrayLike:
        return numpy.load(f"{CACHE_DIR}/{key}.npy", allow_pickle=True)

    def store_np(self, key: str, data: ArrayLike):
        file_path = CACHE_DIR / f"{key}.npy"
        np.save(file_path, data, allow_pickle=True)
        self._lock[key] = self.sha256sum(file_path)
        self._store_lock()

    def _store_lock(self):
        with open(LOCK_FILE, 'w') as f:
            json.dump(self._lock, f)

    @staticmethod
    def sha256sum(filename):
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(filename, 'rb', buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        return h.hexdigest()
