from rs import rate_scale, users, supp_item
import logging
import math
import numpy as np


def sim_rate(rate1, rate2):
    return math.exp(-abs(rate1 - rate2) / rate_scale)


def score(aux, record, ratings):
    item_ids = list(np.nonzero(aux)[0])
    sum = 0
    for item_id in item_ids:
        weight = 1 / supp_item(ratings[:, item_id])
        sum += weight * sim_rate(aux[item_id], record[item_id])
    return sum


def de_anonymization(aux, eccen, ratings):
    candidates = []
    for i in range(len(users)):
        record = ratings[i, :]
        candidates.append(score(aux, record, ratings))
    temp = candidates.copy()
    std = np.std(temp)
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    if (max1 - max2) / std < eccen:
        return None
    else:
        return ratings[candidates.index(max1), :]


def entropic_de(aux, ratings):
    candidates = []
    for i in range(len(users)):
        record = ratings[i, :]
        candidates.append(score(aux, record, ratings))
    std = np.std(candidates)
    temp = [i / std for i in candidates]
    total = sum(temp)
    c = 1 / total
    return (c, temp)
