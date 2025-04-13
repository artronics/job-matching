import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm.auto import tqdm


class JobBertSkillExtraction:
    model = None

    def __init__(self, batch_size: int = 8):
        self.model = SentenceTransformer("jjzha/jobbert_skill_extraction")
        self.batch_size = batch_size

    def encode(self, texts):
        print("Encoding...")
        sorted_indices = np.argsort([len(text) for text in texts])
        sorted_texts = [texts[i] for i in sorted_indices]

        embeddings = []
        for i in tqdm(range(0, len(sorted_texts), self.batch_size)):
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
