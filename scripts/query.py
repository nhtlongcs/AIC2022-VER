import faiss
import numpy as np

# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
dimension = 512

res = faiss.StandardGpuResources()  # use a single GPU
faiss_pool = faiss.IndexFlatIP(dimension)
faiss_pool = faiss.index_cpu_to_gpu(res, 0, faiss_pool)


def compute_faiss(
    queries_embedding,
    gallery_embedding,
    queries_ids,
    targets_ids,
    gallery_ids,
    top_k=10,
):
    """
    Compute score for each metric and return using faiss
    queries_embedding (numpy.ndarray): (n_queries, dimension)
    gallery_embedding (numpy.ndarray): (n_gallery, dimension)
    """

    faiss_pool.reset()
    faiss_pool.add(gallery_embedding)
    top_k_scores_all, top_k_indexes_all = faiss_pool.search(queries_embedding, k=top_k)

    for idx, (top_k_scores, top_k_indexes) in enumerate(
        zip(top_k_scores_all, top_k_indexes_all)
    ):

        current_id = queries_ids[idx]
        target_ids = targets_ids[idx]
        if not isinstance(target_ids, np.ndarray):
            target_ids = np.array([target_ids])

        pred_ids = gallery_ids[top_k_indexes]
        pred_ids = pred_ids.tolist()

        if save_results:
            results_dict[current_id] = {
                "pred_ids": pred_ids,
                "target_ids": target_ids,
                "scores": top_k_scores,
            }
