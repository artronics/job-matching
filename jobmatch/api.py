from dataclasses import asdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sentence_transformers.util import cos_sim

from jobmatch import DATA_DIR
from jobmatch.cache import Cache
from jobmatch.llm import JobBertV2Local
from jobmatch.models import Claimant, Matching, TitleMatchingResponse

app = FastAPI()
cache = Cache()


model = JobBertV2Local(batch_size=8)

# Loading SOC codes and calculating embeddings. This process only needs to be done once
# soc_ds = datasets.make_soc_job_titles().dataset
# soc_titles = np.array([t.get_job_title().title for t in soc_ds])
# soc_embeddings = make_embeddings(model, f"soc_only_titles", soc_titles)
soc_embeddings = []
soc_titles = []


def _match(text: str) -> [(str, float)]:
    """Given an input `text`, we sort all the SOC codes based on similarity to the text"""
    text_emb = model.encode([text])[0]
    similarities = cos_sim(text_emb, soc_embeddings)[0].numpy()
    return sorted(zip(soc_titles, similarities), key=lambda x: x[1], reverse=True)




@app.get("/match")
async def match(text: str, limit: int = 10):
    similarities = _match(text)[:limit]

    titles = [item[0] for item in similarities]
    scores = [float(item[1]) for item in similarities]  # We need to convert float32 to float

    matching = Matching(titles, scores)
    return TitleMatchingResponse(text, matching)


@app.post("/claimant")
async def claimant(claimant: Claimant):
    t = ["foo", "bar"]
    s = [2.3, 3.3]
    m = Matching(t, s)
    tr = TitleMatchingResponse("org", m)
    tr1 = TitleMatchingResponse("org", m)
    ss = {**asdict(tr), **asdict(tr1)}

    updated = claimant.update_socs(ss)

    return updated


@app.get("/claimant-file")
async def download_claimant_file():
    filename = "claimants.zip"

    return FileResponse(DATA_DIR / filename, media_type='application/octet-stream', filename=filename)


@app.get("/")
async def root():
    return {"message": "ok"}


if __name__ == "__main__":
    port = 8080
    print(f"Starting the server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
