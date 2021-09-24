from __future__ import absolute_import, division, print_function

import numpy as np


def evaluate_rec_ndcg_mrr_batch(ratings,results, top_k=10, row_target_position=0):
    ratings = np.array(ratings)
    ratings = ratings[~ np.any(np.isnan(ratings), -1)]

    num_rows = len(ratings)
    if num_rows == 0:
        return 0, 0, 0

    ranks = np.argsort(np.argsort(-np.array(ratings), axis=-1), axis=-1)[:, row_target_position] + 1

    results[2] += np.sum(1 / ranks)

    ranks = ranks[ranks <= top_k]

    results[0] += len(ranks)
    results[1] += np.sum(1 / np.log2(ranks + 1))
