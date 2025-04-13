import json

import numpy as np
from numpy.typing import NDArray

from jobmatch import DATA_DIR
from jobmatch.cache import Cache
from jobmatch.job import JobTitleClassificationPath


class SocCodes:
    def __init__(self, dataset: NDArray[JobTitleClassificationPath]):
        self.dataset = dataset

    def get_job_titles(self) -> NDArray[JobTitleClassificationPath]:
        return self.dataset


def load_soc_codes() -> SocCodes:
    cache = Cache()
    ds_name = "soc_job_titles"
    if cache.exists(ds_name):
        titles = cache.load_np(ds_name)
        return SocCodes(titles)

    soc_titles = DATA_DIR / "job_titles.json"
    with open(soc_titles) as f:
        job_titles = np.array(json.load(f))
        unique_titles = np.unique(job_titles)

        sorted_indices = np.argsort([len(title) for title in unique_titles])
        sorted_texts = [unique_titles[i] for i in sorted_indices]

        titles = [JobTitleClassificationPath([t]) for t in sorted_texts if len(t) > 3]
        cache.store_np(ds_name, titles)

        return SocCodes(np.array(titles))


if __name__ == '__main__':
    soc_codes = load_soc_codes()
    print(soc_codes.get_job_titles())
