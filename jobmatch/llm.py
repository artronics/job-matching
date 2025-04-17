import pathlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm.auto import tqdm

prj_path = pathlib.Path(__file__).parent.parent.resolve()

class BaseModel:
    name: str
    model = None
    batch_size: int

    def __init__(self, name: str, batch_size: int = 8):
        self.name = name
        self.model = SentenceTransformer(name)
        self.batch_size = batch_size

    def encode(self, texts, name: str = ""):
        if len(texts) > 10:
            print(f"Encoding {name} with {len(texts)} item(s)...")
        sorted_indices = np.argsort([len(text) for text in texts])
        sorted_texts = [texts[i] for i in sorted_indices]

        embeddings = []
        for i in tqdm(range(0, len(sorted_texts), self.batch_size),  disable=(len(sorted_texts)<10)):
            batch = sorted_texts[i:i + self.batch_size]
            embeddings.append(self._encode_batch(batch))

        sorted_embeddings = np.concatenate(embeddings)
        original_order = np.argsort(sorted_indices)

        return sorted_embeddings[original_order]

    def _encode_batch(self, texts):
        features = self.model.tokenize(texts)
        features = batch_to_device(features, self.model.device)
        features["text_keys"] = ["anchor"]
        with torch.no_grad():
            out_features = self.model.forward(features)

        return out_features["sentence_embedding"].cpu().numpy()

    def __str__(self):
        return self.name


class JobBertV2(BaseModel):
    def __init__(self, batch_size: int = 8):
        super().__init__("TechWolf/JobBERT-v2", batch_size)

class JobBertV2Local(BaseModel):
    def __init__(self, batch_size: int = 8):
        path_to_model = str(prj_path / "models" / "JobBERT-v2")
        super().__init__(path_to_model, batch_size)



class ContextSkillExtraction(BaseModel):
    def __init__(self, batch_size: int = 8):
        # super().__init__("TechWolf/ConTeXT-Skill-Extraction-base", batch_size)
        # The above code doesn't work. It complaints about missing `model_type`. You should clone the repo
        # and use your local machine path to solve the issue. I'm not sure why this happens.
        # The project has `model_type: mpnet` which is supported by `sentence_transformers`
        super().__init__("/Users/jalal/projects/llm/ConTeXT-Skill-Extraction-base", batch_size)
        # Other models based on this one has the same issue
        # super().__init__("abd1987/esco-context-skill-extraction", batch_size)
        # super().__init__("jensjorisdecorte/ConTeXT-Skill-Extraction-base", batch_size)
