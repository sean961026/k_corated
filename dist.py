import math
from rs import rating_scale
import logging


def hellinger_distance(p, q):
    s = 0
    zipped = zip(p, q)
    for pi, qi in zipped:
        s += math.sqrt(pi * qi)
    distance = math.sqrt(1 - s)
    return distance


def get_prop_dist_from_ratings(ratings_list):
    p = [0] * rating_scale
    for rating in ratings_list:
        p[int(rating) - 1] += 1
    s = sum(p)
    return [i / s for i in p]
