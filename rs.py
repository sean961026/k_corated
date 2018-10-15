import numpy as np
import logging
import pandas as pd
import argparse
import os
import math

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
np.seterr(all='raise')
directory = 'ml-100k/'
min_rating = 1
max_rating = 5
rating_scale = max_rating - min_rating
user_size = 943
item_size = 1682
mode_choices = ['all', 'distance_co', 'correlation_co', 'trust_co', 'cos_co', 'tanimoto_co']
dataset_choices = ['u', 'u1', 'u2', 'u3', 'u4', 'u5']
unknown_rating = 0
unknown_weight = 99


def get_ratings_from_jester():
    filename = 'jester_ratings.csv'
    if os.path.exists(filename):
        original_ratings = load(filename)
    else:
        original_ratings = np.zeros(shape=(user_size, item_size))
        with open('jester_dataset_2/jester_ratings.dat', 'r') as file:
            lines = file.readlines()
            for line in lines:
                user_id, item_id, rating = line.split('\t\t')
                original_ratings[int(user_id) - 1, int(item_id) - 1] = float(rating) + 11
        dump(filename, original_ratings)
    return original_ratings


def get_ratings_from_ml_100k(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'
    if os.path.exists(filename):
        original_ratings = load(filename)
    else:
        original_ratings = np.zeros(shape=(user_size, item_size))
        with open(directory + filename[:-4], 'r') as file:
            lines = file.readlines()
            for line in lines:
                user_id, item_id, rating, timestamp = line.split('\t')
                original_ratings[int(user_id) - 1, int(item_id) - 1] = float(rating)
        dump(filename, original_ratings)
    return original_ratings


def supp_user(user):
    user = np.array(user) - unknown_rating
    return set(np.nonzero(user)[0])


def supp_item(item):
    item = np.array(item) - unknown_rating
    return set(np.nonzero(item)[0])


def co_rated_items(user_1, user_2):
    return list(supp_user(user_1).intersection(supp_user(user_2)))


def co_rated_part(user_1, user_2):
    u1 = []
    u2 = []
    for i in co_rated_items(user_1, user_2):
        u1.append(user_1[i])
        u2.append(user_2[i])
    return u1, u2


def co_bought_users(item1, item2):
    return list(supp_item(item1).intersection(supp_item(item2)))


def mean(user):
    rated_items = supp_user(user)
    s = 0
    for i in rated_items:
        s += user[i]
    return s / len(rated_items)


def weight(user_1, user_2, mode, co_threshold):
    if mode == 'distance_co':
        w = weight_distance_co(user_1, user_2, co_threshold)
    elif mode == 'correlation_co':
        w = weight_correlation_co(user_1, user_2, co_threshold)
    elif mode == 'cos_co':
        w = weight_cos_co(user_1, user_2, co_threshold)
    elif mode == 'tanimoto_co':
        w = weight_tanimoto_co(user_1, user_2, co_threshold)
    elif mode == 'trust_co':
        w = weight_trust_co(user_1, user_2, co_threshold)
    else:
        raise ValueError
    return w


def weight_distance_co(user_1, user_2, threshold):
    corated = co_rated_items(user_1, user_2)
    if len(corated) < threshold:
        return unknown_weight
    dis = 0
    for i in corated:
        dis += (user_1[i] - user_2[i]) ** 2
    w = 1 / (1 + math.sqrt(dis))
    if w < 0 or w > 1:
        return unknown_weight
    else:
        return w


def weight_correlation_co(user_1, user_2, threshold):
    u1, u2 = co_rated_part(user_1, user_2)
    if len(u1) < threshold:
        return unknown_weight
    try:
        w = np.corrcoef(u1, u2)[0, 1]
        if w < 0 or w > 1:
            return unknown_weight
    except:
        w = unknown_weight
    return w


def weight_cos_co(user_1, user_2, threshold):
    u1, u2 = co_rated_part(user_1, user_2)
    if len(u1) < threshold:
        return unknown_weight
    dot = np.matmul(u1, u2)
    norm_1 = np.linalg.norm(u1)
    norm_2 = np.linalg.norm(u2)
    try:
        w = dot / (norm_1 * norm_2)
        if w < 0 or w > 1:
            return unknown_weight
    except:
        w = unknown_weight
    return w


def weight_tanimoto_co(user_1, user_2, threshold):
    u1, u2 = co_rated_part(user_1, user_2)
    if len(u1) < threshold:
        return unknown_weight
    dot = np.matmul(u1, u2)
    norm_1 = np.linalg.norm(u1)
    norm_2 = np.linalg.norm(u2)
    try:
        w = dot / (norm_1 ** 2 + norm_2 ** 2 - dot)
        if w < 0 or w > 1:
            return unknown_weight
    except:
        w = unknown_weight
    return w


def weight_trust_co(user_1, user_2, threshold):
    u1, u2 = co_rated_part(user_1, user_2)
    if len(u1) < threshold:
        return unknown_weight
    mean_1 = mean(user_1)
    mean_2 = mean(user_2)
    z = zip(u1, u2)
    s = 0
    for r1, r2 in z:
        predict_rating = r1 - mean_1 + mean_2
        difference = predict_rating - r2
        trust_percent = 1 - abs(difference) / rating_scale
        s += trust_percent
    w = s / len(u1)
    if w < 0 or w > 1:
        return unknown_weight
    else:
        return w


def neareast_neighbors_by_threshold(user_id, web, threshold):
    weights = web[user_id, :]
    neighbors = []
    for i in range(user_size):
        if weights[i] != unknown_weight and weights[i] > threshold and i != user_id:
            neighbors.append(i)
    return neighbors


def nearest_neighbors_by_fix_number(user_id, web, n):
    weights = web[user_id, :]
    neighbors = []
    for i in range(user_size):
        if weights[i] != unknown_weight and i != user_id:
            neighbors.append(i)
    if len(neighbors) < n:
        return neighbors
    else:
        neighbors.sort(reverse=True)
        return [neighbors[i] for i in range(n)]


def create_web(original_ratings, mode, threshold):
    web = np.zeros(shape=(user_size, user_size))
    for i in range(user_size):
        for j in range(user_size):
            user_1 = original_ratings[i, :]
            user_2 = original_ratings[j, :]
            web[i, j] = weight(user_1, user_2, mode, threshold) if i != j else 1
    return web


def dump(filename, matrix):
    if not filename.endswith('.csv'):
        filename += '.csv'
    logging.info('dumping %s', filename)
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)


def load(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'
    logging.info('loading %s', filename)
    matrix = np.loadtxt(filename, delimiter=',')
    return matrix


def pd_rating(original_ratings, user_id, item_id, web, neighbors):
    user = original_ratings[user_id, :]
    user_mean = mean(user)
    up = 0
    down = 0
    for neighbor_id in neighbors:
        neighbor = original_ratings[neighbor_id, :]
        neighbor_mean = mean(neighbor)
        neighbor_rating = neighbor[item_id]
        diff = neighbor_rating - neighbor_mean
        weight = web[user_id, neighbor_id]
        if neighbor_rating != unknown_rating and weight != unknown_weight:
            up += weight * diff
            down += weight
    try:
        predicted_rating = user_mean + up / down
        if predicted_rating > max_rating:
            predicted_rating = max_rating
        elif predicted_rating < min_rating:
            predicted_rating = min_rating
    except:
        predicted_rating = user_mean
    return predicted_rating


def main():
    # will create
    # [dataset].base.csv
    # [mode]_[threshold].csv
    parser = argparse.ArgumentParser(description='Create ratings and webs of a certain file')
    parser.add_argument('-d', '--dataset', choices=dataset_choices)
    parser.add_argument('-t', '--threshold', type=int)
    parser.add_argument('-m', '--mode', choices=mode_choices)
    args = parser.parse_args()
    dataset = args.dataset
    original_ratings = get_ratings_from_ml_100k(dataset + '.base')
    mode = args.mode
    co_threshold = args.threshold
    if mode == 'all':
        for mode in mode_choices:
            if mode != 'all':
                web = create_web(original_ratings, mode, co_threshold)
                filename = mode + '_' + str(co_threshold)
                dump(filename, web)
    else:
        web = create_web(original_ratings, mode, co_threshold)
        filename = mode + '_' + str(co_threshold)
        dump(filename, web)


if __name__ == '__main__':
    main()
