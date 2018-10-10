import math
from rs import supp_item, rate_scale
import logging


def hellinger_distance(p, q):
    s = 0
    zipped = zip(p, q)
    for pi, qi in zipped:
        s += math.sqrt(pi * qi)
    return math.sqrt(1 - s)


def get_prop_dist_global(item_record):
    p = [0] * rate_scale
    rated_users = supp_item(item_record)
    for user_id in rated_users:
        rating = int(item_record[user_id])
        try:
            p[rating - 1] += 1
        except:
            logging.exception(item_record[user_id])
        s = sum(p)
    return [i / s for i in p]


def get_prop_dist_local(item_record, local):
    p = [0] * rate_scale
    rated_users = supp_item(item_record)
    for user_id in rated_users:
        if user_id in local:
            rating = int(item_record[user_id])
            try:
                p[rating - 1] += 1
            except:
                logging.exception(item_record[user_id])
    s = sum(p)
    return [i / s for i in p]
