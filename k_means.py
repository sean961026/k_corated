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
    def __init__(self, original_ratings, centroid_index):
        self.centroid = normalize(original_ratings[centroid_index, :])
        self.original_ratings = original_ratings
        self.points = [centroid_index]

    def _update_centroid(self, point):
        point_vec = normalize(self.original_ratings[point, :])
        for i in range(len(point_vec)):
            if self.centroid[i] == 0 and point_vec[i] == 1:
                self.centroid[i] = 1

    def add_new_point(self, point):
        self.points.append(point)
        self._update_centroid(point)

    def distance_to(self, point):
        return (self._cost_of(point) + 1) / (self._corated_of(point) + 1)

    def _cost_of(self, point):
        new = 0
        point_vec = normalize(self.original_ratings[point, :])
        for i in range(len(point_vec)):
            if self.centroid[i] == 1 and point_vec[i] == 0:
                new += 1
        return new * len(self.points)

    def _corated_of(self, point):
        corated = 0
        point_vec = normalize(self.original_ratings[point, :])
        for i in range(len(point_vec)):
            if self.centroid[i] == 1 and point_vec[i] == 1:
                corated += 1
        return corated

    def dis_sum(self):
        s = sum(self.centroid) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(normalize(self.original_ratings[point, :]))
        return s - t


def dis_of_clusters(clusters):
    s = 0
    for cluser in clusters:
        s += cluser.dis_sum()
    return s


def k_means_simple(original_ratings, k, mode):
    all_points = [i for i in range(original_ratings.shape[0])]
    seeds = get_initial_seeds(original_ratings, k, mode)
    clusters = [Cluster(original_ratings, seed) for seed in seeds]
    for seed in seeds:
        all_points.remove(seed)
    while len(all_points) != 0:
        for cluster in clusters:
            distances = [cluster.distance_to(point) for point in all_points]
            which = all_points[distances.index(min(distances))]
            cluster.add_new_point(which)
            all_points.remove(which)
    logging.info('the dis of such clusters is %d', dis_of_clusters(clusters))
    return clusters


def k_means(original_ratings, k, mode):
    best_clusters = k_means_simple(original_ratings, k, mode)
    for i in range(100):
        temp_clusters = k_means_simple(original_ratings, k, mode)
        if dis_of_clusters(best_clusters) > dis_of_clusters(temp_clusters):
            best_clusters = temp_clusters
    return best_clusters
