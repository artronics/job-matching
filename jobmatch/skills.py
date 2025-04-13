import numpy as np
import pandas as pd
from numpy.typing import NDArray

from jobmatch import DATA_DIR
from jobmatch.cache import Cache


class SkillsDataset:
    def __init__(self, dataset: NDArray):
        self.dataset = dataset


def make_onet_skills() -> SkillsDataset:
    cache = Cache()
    ds_name = "onet_only_skills"
    if cache.exists(ds_name):
        skills = cache.load_np(ds_name)
        return SkillsDataset(skills)

    with open(DATA_DIR / "onet_skills.csv") as f:
        skills = pd.read_csv(f).to_numpy()
        unique_skills = np.unique(skills)

        sorted_indices = np.argsort([len(title) for title in unique_skills])
        sorted_skills = [unique_skills[i] for i in sorted_indices]

        cache.store_np(ds_name, sorted_skills)

        return SkillsDataset(np.array(sorted_skills))


if __name__ == '__main__':
    s = make_onet_skills()
    pass
