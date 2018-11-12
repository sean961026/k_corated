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

    def update_centroid(self):
        size = len(self.centroid)
        temp = self._get_items_sum()
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

    def _get_items_sum(self):
        temp = [0] * Cluster.original_ratings.shape[1]
        for i in range(Cluster.original_ratings.shape[1]):
            for point in self.points:
                temp[i] += 0 if Cluster.original_ratings[point, i] == 0 else 1
        return temp

    def _get_corated(self):
        temp = self._get_items_sum()
        return normalize(temp)

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
        s = sum(self._get_corated()) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(normalize(Cluster.original_ratings[point, :]))
        return s - t

    def new_dis(self):
        n = sum(self.centroid)
        temp = self._get_items_sum()
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])
        top_temp = [sorted_temp[i] for i in range(n)]
        top_index = [z[1] for z in top_temp]
        s = 0
        for index in top_index:
            s += temp[index]
        all_s = n * len(self.points)
        return all_s - s

    def cost(self):
        n = sum(self.centroid)
        temp = self._get_items_sum()
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])
        top_temp = [sorted_temp[i] for i in range(n)]
        top_index = [z[1] for z in top_temp]
        cost = 0
        for i in range(len(temp)):
            if i not in top_index:
                cost += temp[i]
        return cost

    def info(self):
        point_size = len(self.points)
        dis = self.dis_sum()
        corated = self._get_corated()
        temp = self._get_items_sum()
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])

        def top_n_contribution(n):
            top_temp = [sorted_temp[i] for i in range(n)]
            top_index = [z[1] for z in top_temp]
            s = 0
            for index in top_index:
                s += temp[index]
            all_s = n * len(self.points)
            cost = 0
            for i in range(len(temp)):
                if i not in top_index:
                    cost += temp[i]
            return all_s - s, cost

        centroid_contribution, cost_1 = top_n_contribution(sum(self.centroid))
        nine_contribution, cost_2 = top_n_contribution(int(sum(corated) * 0.9))
        eight_contribution, cost_3 = top_n_contribution(int(sum(corated) * 0.8))
        info = {'point_size': point_size, 'centroid_size': sum(self.centroid), 'corated_size': sum(corated), 'dis': dis,
                'centroid_portion': centroid_contribution, '90%_portion': nine_contribution,
                '80%_portion': eight_contribution, 'centroid_cost': cost_1, '90%_cost': cost_2, '80%_cost': cost_3}
        logging.info(info)


def dis_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        # s += cluster.dis_sum()
        s += cluster.new_dis()
    return s


def cost_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        # s += cluster.dis_sum()
        s += cluster.cost()
    return s


def k_means_iter_once(clusters):
    point_size = Cluster.original_ratings.shape[0]
    for point in range(point_size):
        dis = []
        for cluster in clusters:
            dis.append(cluster.distance_to(point))
        min_cluster = dis.index(min(dis))
        clusters[min_cluster].add_new_point(point)
    logging.info('the dis of such clusters is %d', dis_of_clusters(clusters))
    logging.info('the cost of such clusters is %d', cost_of_clusters(clusters))
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
        clear_all(clusters)
    k_means_iter_once(clusters)
    logging.info('the dis of such clusters is %d', dis_of_clusters(clusters))
    return clusters
