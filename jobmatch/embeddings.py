from numpy.typing import NDArray

from jobmatch.cache import Cache
from jobmatch.llm import BaseModel

cache = Cache()


def make_embeddings(model: BaseModel, name: str, data: NDArray):
    """Make embeddings for the data and cache the result.
    Use it only if you want the cache otherwise, just use `encode` method of the model."""

    embeddings_name = f"embeddings_{model.name.replace('/', '--')}__{name}"
    if cache.exists(embeddings_name):
        return cache.get(embeddings_name)

    embeddings = model.encode(data, name)
    cache.set(embeddings_name, embeddings)

    return embeddings
