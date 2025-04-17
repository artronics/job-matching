import json
import sys
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers.util import cos_sim
from tqdm.auto import tqdm

from jobmatch import job_datasets as datasets, DATA_DIR
from jobmatch.cache import Cache
from jobmatch.embeddings import make_embeddings
from jobmatch.llm import JobBertV2Local
from jobmatch.models import Matching, TitleMatchingResponse

cache = Cache()
model = JobBertV2Local(batch_size=8)

# Loading SOC codes and calculating embeddings. This process only needs to be done once
soc_ds = datasets.make_soc_job_titles().dataset
soc_titles = np.array([t.get_job_title().title for t in soc_ds])
soc_embeddings = make_embeddings(model, f"soc_only_titles", soc_titles)


def match_text_to_soc(text: str, limit: int) -> ([str], [float]):
    """Given an input `text`, we sort all the SOC codes based on similarity to the text"""
    text_emb = model.encode([text])[0]
    similarities = cos_sim(text_emb, soc_embeddings)[0].numpy()
    similarities = sorted(zip(soc_titles, similarities), key=lambda x: x[1], reverse=True)[:limit]

    titles = [item[0] for item in similarities]
    scores = [float(item[1]) for item in similarities]  # We need to convert float32 to float
    return titles, scores



class Item:
    jobTitle: List[dict]
    preferredJobs: List[dict]

    def __init__(self, job_title, preferred_jobs):
        self.jobTitle = job_title
        self.preferredJobs = preferred_jobs

    def to_dict(self):
        return {
            "jobTitle": self.jobTitle,
            "preferredJobs": self.preferredJobs
        }

    def to_json(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)


def process(filename) -> List[Item]:
    claimants = pd.read_csv(DATA_DIR / filename, encoding="utf-8", header=1)

    socs = []
    matching_limit = 5
    count = 0
    for _, claimant in tqdm(claimants.iterrows(), total=len(claimants)):
        count += 1
        jobTitles = claimant[0]
        preferredJobs = claimant[1]

        jt = {}
        if isinstance(jobTitles, str):
            for jobTitle in jobTitles.split(','):
                titles, scores = match_text_to_soc(jobTitle, matching_limit)
                matching = Matching(titles, scores)
                jt.update(TitleMatchingResponse(jobTitle, matching).as_dict())

        pj = {}
        if isinstance(preferredJobs, str):
            for preferredJob in preferredJobs.split(','):
                titles, scores = match_text_to_soc(preferredJob, matching_limit)
                matching = Matching(titles, scores)
                pj.update(TitleMatchingResponse(preferredJob, matching).as_dict())

        socs.append(Item(jt, pj))

    return socs


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "claimant.csv"
    socs = process(filename)

    with open("claimant.jsonl", "w") as f:
        for item in socs:
            json_line = json.dumps(item.to_dict())
            f.write(json_line + '\n')


if __name__ == '__main__':
    main()
