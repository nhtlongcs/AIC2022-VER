from typing import Dict, List, Any
import faiss
import numpy as np
from src.utils.device import detach

from . import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class Accuracy:
    """
    Compute the accuracy of the model.
    Expect the model to return a dict with the following keys:
    - "pairs": a tuple of two torch.tensors, each of shape (N, D), 
    where N is the number of pairs and D is the embedding dimension.
    Each pair is a pair of visual and language embeddings. Have a unique id for each pair.
    """

    def __init__(self, dimension=768, topk=(1,)):
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        self.topk = topk
        self.faiss_pool = faiss.IndexFlatIP(dimension)
        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            self.faiss_pool = faiss.index_cpu_to_all_gpus(self.faiss_pool)
            print(f"Using {ngpus} GPU to evaluate")
        else:
            print("Using CPU to evaluate")
        self.reset()

    def similarity_search(self, queries_embedding, gallery_embedding, top_k=10):
        """
        Compute the similarity between queries and gallery embeddings.
        """

        self.faiss_pool.reset()
        self.faiss_pool.add(gallery_embedding)
        top_k_scores_all, top_k_indexes_all = self.faiss_pool.search(
            queries_embedding, k=top_k
        )
        return top_k_scores_all, top_k_indexes_all

    def update(self, output: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        pairs  = detach(output["pairs"])
        visual_embeddings, lang_embeddings = (
            pairs[0].cpu().numpy(),
            pairs[1].cpu().numpy(),
        )
        self.visual_embeddings.append(visual_embeddings)
        self.lang_embeddings.append(lang_embeddings)

        self.sample_size += visual_embeddings.shape[0] # batchsize

    def calculate(self, **kwargs):
        lang_embeddings = np.concatenate(self.lang_embeddings, axis=0)
        visual_embeddings = np.concatenate(self.visual_embeddings, axis=0)

        target_ids = np.array([i for i in range(self.sample_size)])
        gallery_ids = np.array([i for i in range(self.sample_size)])

        top_k_scores_all, top_k_indexes_all = self.similarity_search(
            queries_embedding=lang_embeddings,
            gallery_embedding=visual_embeddings,
            top_k=max(self.topk),
        )

        self.correct = {k: 0 for k in self.topk}
        for idx, (top_k_scores, top_k_indexes) in enumerate(
            zip(top_k_scores_all, top_k_indexes_all)
        ):

            for k in self.topk:
                avail_ids = np.where(top_k_indexes != -1)
                pred_ids = gallery_ids[top_k_indexes[avail_ids][:k]].tolist()
                correct = (pred_ids == target_ids[idx]).sum()
                self.correct[k] += correct

    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.sample_size = 0
        self.visual_embeddings = []
        self.lang_embeddings = []

    def value(self):
        metric_dict = {k: self.correct[k] / self.sample_size for k in self.topk}
        metric_score = sum(metric_dict[k] for k in self.topk) / len(
            self.topk
        )  # Average of top-k accuracy
        return {"score": metric_score, "score_dict": metric_dict}

    def summary(self):
        result_dict = self.value()
        print(f"Average accuracy: {result_dict['score']}")
        print(f"Accuracy: {result_dict['score_dict']}")

