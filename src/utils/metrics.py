import numpy as np

def hit_ratio(recommends, gt_item):
    if gt_item in recommends:
        return 1
    return 0

def ndcg(recommends, gt_item):
    if gt_item in recommends:
        index = recommends.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0
