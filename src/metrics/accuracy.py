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

    def calculate(self, output, **kwargs):
        pairs = detach(output["pairs"])
        target_ids = np.array(range(len(pairs[0])))
        gallery_ids = np.array(range(len(pairs[0])))

        visual_embeddings, lang_embeddings = (
            pairs[0].cpu().numpy(),
            pairs[1].cpu().numpy(),
        )

        top_k_scores_all, top_k_indexes_all = self.similarity_search(
            queries_embedding=lang_embeddings,
            gallery_embedding=visual_embeddings,
            top_k=max(self.topk),
        )
        res = {k: 0 for k in self.topk}
        for idx, (top_k_scores, top_k_indexes) in enumerate(
            zip(top_k_scores_all, top_k_indexes_all)
        ):

            for k in self.topk:
                avail_ids = np.where(top_k_indexes != -1)
                pred_ids = gallery_ids[top_k_indexes[avail_ids][:k]].tolist()
                correct = (pred_ids == target_ids[idx]).sum()
                res[k] += correct
        return res, len(pairs[0])

    def update(self, value):
        for k in value[0].keys():
            self.correct[k] += value[0][k]
        self.sample_size += value[1]

    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.sample_size = 0.0

    def value(self):
        metric_dict = {k: self.correct[k] / self.sample_size for k in self.topk}
        metric_score = sum(metric_dict[k] for k in self.topk) / len(
            self.topk
        )  # Average of top-k accuracy
        return {"score": metric_score, "score_dict": metric_dict}

    def summary(self):
        print(f"Average accuracy: {self.value()['score']}")
        print(f"Accuracy: {self.value()['score_dict']}")

