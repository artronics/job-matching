import numpy as np
from numpy.typing import NDArray

from jobmatch.cache import Cache
from jobmatch.job import JobTitleClassificationPath


class Advert:
    title: JobTitleClassificationPath
    content: [str]

    def __init__(self, title: JobTitleClassificationPath, content: str):
        self.title = title
        self.content = content


class AdvertDataset:
    def __init__(self, dataset: NDArray[Advert]):
        self.dataset = dataset

    def choice(self) -> Advert:
        return np.random.choice(self.dataset)


def load_dataset() -> AdvertDataset:
    import kagglehub
    import pandas as pd

    cache = Cache()

    ds_name = "armenian-online-job-postings"
    if cache.exists(ds_name):
        adverts = cache.load_np(ds_name)
        AdvertDataset(adverts)

    path = kagglehub.dataset_download(f"udacity/{ds_name}")
    df = pd.read_csv(f"{path}/online-job-postings.csv")

    adverts = []
    df["title"] = df["Title"].fillna("").astype(str)
    df["jobDescription"] = df["JobDescription"].fillna("").astype(str)
    df["jobRequirement"] = df["JobRequirment"].fillna("").astype(str)

    for _, row in df.iterrows():
        content = f"{row['jobDescription']}\n{row['jobRequirement']}"
        title = JobTitleClassificationPath([row["title"]])
        advert = Advert(title, content)
        adverts.append(advert)

    adverts = np.array(adverts)
    cache.store_np(ds_name, adverts)

    return AdvertDataset(adverts)


if __name__ == '__main__':
    s = load_dataset()
