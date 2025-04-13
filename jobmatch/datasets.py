import json
from dataclasses import dataclass

import kagglehub
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from jobmatch import DATA_DIR
from jobmatch.cache import Cache
from jobmatch.models import JobTitle, Advert, JobTitleClassificationPath

cache = Cache()


def _sort_and_cleanup(data: NDArray) -> NDArray:
    unique = np.unique(data)
    sorted_indices = np.argsort([len(item) for item in unique])
    length_sorted = [unique[i] for i in sorted_indices]
    return np.array(length_sorted)


@dataclass
class BaseDataset:
    dataset: NDArray
    name: str

    def __init__(self, name: str, dataset: NDArray):
        self.name = name
        self.dataset = dataset

    def choice(self):
        return np.random.choice(self.dataset)


def make_onet_skills_dataset() -> BaseDataset:
    ds_name = "ds_onet_skills"
    if cache.exists(ds_name):
        skills = cache.load_np(ds_name)
        return BaseDataset(ds_name, skills)

    with open(DATA_DIR / "onet_skills.csv") as f:
        skills = pd.read_csv(f).to_numpy()
        cleaned_ds = _sort_and_cleanup(skills)
        cache.store_np(ds_name, cleaned_ds)

        return BaseDataset(ds_name, cleaned_ds)


def make_job_adverts_dataset() -> BaseDataset:
    ds_name = "ds_armenian_online_job_postings"
    if cache.exists(ds_name):
        adverts = cache.load_np(ds_name)
        BaseDataset(ds_name, adverts)

    path = kagglehub.dataset_download(f"udacity/{ds_name}")
    df = pd.read_csv(f"{path}/online-job-postings.csv")

    df["title"] = df["Title"].fillna("").astype(str)
    df["jobDescription"] = df["JobDescription"].fillna("").astype(str)
    df["jobRequirement"] = df["JobRequirment"].fillna("").astype(str)

    adverts = []
    for _, row in df.iterrows():
        title = row["title"]
        contents = [row['jobDescription'], row['jobRequirement']]
        adverts.append(Advert(JobTitle(title), contents))

    adverts = np.array(adverts)
    cache.store_np(ds_name, adverts)

    return BaseDataset(ds_name, adverts)


def make_soc_job_titles() -> BaseDataset:
    ds_name = "ds_soc_job_titles"
    if cache.exists(ds_name):
        titles = cache.load_np(ds_name)
        return BaseDataset(ds_name, titles)

    soc_titles = DATA_DIR / "job_titles.json"
    with open(soc_titles) as f:
        job_titles = np.array(json.load(f))
        cleaned_ds = _sort_and_cleanup(job_titles)

        titles = [JobTitleClassificationPath([t]) for t in cleaned_ds if len(t) > 3]
        cache.store_np(ds_name, titles)

        return BaseDataset(ds_name, np.array(titles))
