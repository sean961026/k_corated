import random
import numpy as np
import logging


def get_initial_seeds(original_ratings, size, mode):
    if mode == 'random':
        seeds = get_initial_seeds_randomly(original_ratings, size)
    elif mode == 'other':
        seeds = None
    else:
        raise ValueError
    return seeds


def get_initial_seeds_randomly(original_ratings, random_size):
    user_size = original_ratings.shape[0]
    if user_size < random_size:
        raise ValueError
    else:
        all = [i for i in range(user_size)]
        return random.sample(all, random_size)


def normalize(vec):
    data = []
    for i in vec:
        element = 0 if i == 0 else 1
        data.append(element)
    return np.array(data)


class Cluster:
    original_ratings = None

    def __init__(self, centroid_index):
        self.centroid = normalize(Cluster.original_ratings[centroid_index, :])
        self.points = []
        self.corated = self.centroid

    def update_centroid(self):
        size = len(self.centroid)
        temp = [0] * Cluster.original_ratings.shape[1]
        for i in range(Cluster.original_ratings.shape[1]):
            for point in self.points:
                temp[i] += 0 if Cluster.original_ratings[point, i] == 0 else 1
        self.corated = normalize(temp)
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])
        top_temp = [sorted_temp[i] for i in range(size)]
        top_index = [z[1] for z in top_temp]
        for i in range(Cluster.original_ratings.shape[1]):
            if i not in top_index:
                temp[i] = 0
        self.centroid = normalize(temp)

    def clear(self):
        self.points.clear()

    def add_new_point(self, point):
        self.points.append(point)

    def distance_to(self, point):
        point_vec = normalize(Cluster.original_ratings[point, :])
        corated = 0
        for i in range(len(point_vec)):
            if self.centroid[i] == 1 and point_vec[i] == 1:
                corated += 1
        return -corated / (sum(self.centroid) + sum(point_vec) - corated)

    def dis_sum(self):
        s = sum(self.corated) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(normalize(Cluster.original_ratings[point, :]))
        return s - t


def dis_of_clusters(clusters):
    s = 0
    for cluser in clusters:
        s += cluser.dis_sum()
    return s


def k_means_iter_once(clusters):
    point_size = Cluster.original_ratings.shape[0]
    for point in range(point_size):
        dis = []
        for cluster in clusters:
            dis.append(cluster.distance_to(point))
        min_cluster = dis.index(min(dis))
        clusters[min_cluster].add_new_point(point)
    for cluster in clusters:
        cluster.update_centroid()


def clear_all(clusters):
    for cluster in clusters:
        cluster.clear()


def k_means(original_ratings, k, mode):
    Cluster.original_ratings = original_ratings
    seeds = get_initial_seeds(original_ratings, k, mode)
    clusters = [Cluster(seed) for seed in seeds]
    for i in range(10):
        k_means_iter_once(clusters)
        logging.info('the dis of such clusters is %d', dis_of_clusters(clusters))
        clear_all(clusters)
    k_means_iter_once(clusters)
    logging.info('the dis of such clusters is %d', dis_of_clusters(clusters))
    return clusters
