from typing import Any, Dict, List, Optional

import torch
import numpy as np
from src.utils.device import detach, move_to
from src.utils.faiss_retrieval import FaissRetrieval


class RetrievalMetric:
    """
    Wrapper for computing all retrieval evaluation metrics
    """

    def __init__(self, metrics: List, dimension:int = 768, max_k:int = 30, **kwargs):
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        self.max_k = max_k
        self.metrics = metrics
        self.retriever = FaissRetrieval(dimension=dimension)
        self.reset()

    def update(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        pairs = move_to(detach(output["pairs"]), torch.device('cpu'))
        ids = move_to(batch['car_ids'], torch.device('cpu'))
        visual_embedding, lang_embedding = pairs[0].numpy(), pairs[1].numpy()
        self.gallery_embeddings.append(visual_embedding)
        self.query_embeddings.append(lang_embedding)
        self.all_ids.append(ids)
        self.sample_size += lang_embedding.shape[0]

    def compute(self):
        
        self.query_embeddings = np.concatenate(self.query_embeddings, axis=0)
        self.gallery_embeddings = np.concatenate(self.gallery_embeddings, axis=0)
        self.all_ids = torch.cat(self.all_ids, dim=0).numpy().tolist()

        target_ids = np.array([i for i in range(self.sample_size)])
        gallery_ids = np.array([i for i in range(self.sample_size)])

        top_k_scores_all, top_k_indexes_all = self.retriever.similarity_search(
            query_embeddings=self.query_embeddings,
            gallery_embeddings=self.gallery_embeddings,
            top_k=self.max_k,
            query_ids=self.all_ids, target_ids=self.all_ids, gallery_ids=self.all_ids,
            save_results="temps/query_results.json"
        )

        result_dict = {}
        for top_k_indexes, target_id in zip(top_k_indexes_all, target_ids):
            pred_ids = gallery_ids[top_k_indexes] # gallery id
            for metric in self.metrics:
                metric.update(pred_ids, [target_id])

        for metric in self.metrics:
            result_dict.update(metric.value())

        return result_dict

    def reset(self):
        self.sample_size = 0
        self.gallery_embeddings = []
        self.query_embeddings = []
        self.all_ids = []
        for metric in self.metrics:
            metric.reset()

    def value(self):
        metric_dict = self.compute()
        return metric_dict