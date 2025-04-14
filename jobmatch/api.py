import pathlib

import numpy as np
import uvicorn
from fastapi import FastAPI
from sentence_transformers.util import cos_sim

import jobmatch.job_datasets as datasets
from jobmatch.cache import Cache
from jobmatch.embeddings import make_embeddings
from jobmatch.llm import BaseModel

prj_path = pathlib.Path(__file__).parent.parent.resolve()

app = FastAPI()
cache = Cache()


class JobBertV2Local(BaseModel):
    def __init__(self, batch_size: int = 8):
        path_to_model = str(prj_path / "models" / "JobBERT-v2")
        super().__init__(path_to_model, batch_size)


model = JobBertV2Local(batch_size=8)

# Loading SOC codes and calculating embeddings. This process only needs to be done once
soc_ds = datasets.make_soc_job_titles().dataset
soc_titles = np.array([t.get_job_title().title for t in soc_ds])
soc_embeddings = make_embeddings(model, f"soc_only_titles", soc_titles)


def _match(text: str) -> [(str, float)]:
    """Given an input `text`, we sort all the SOC codes based on similarity to the text"""
    text_emb = model.encode([text])[0]
    similarities = cos_sim(text_emb, soc_embeddings)[0].numpy()
    return sorted(zip(soc_titles, similarities), key=lambda x: x[1], reverse=True)


class TitleMatchResponse:
    """Response object contains **sorted** titles. Scores are provided as a separate list."""
    titles: [str]
    scores: [float]

    def __init__(self, titles: [str], scores: [float]):
        self.titles = titles
        self.scores = scores


@app.get("/match")
async def match(text: str, limit: int = 10):
    similarities = _match(text)[:limit]

    titles = [item[0] for item in similarities]
    scores = [float(item[1]) for item in similarities]  # We need to convert float32 to float

    return TitleMatchResponse(titles, scores)


@app.get("/")
async def root():
    return {"message": "ok"}


if __name__ == "__main__":
    port = 8080
    print(f"Starting the server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
