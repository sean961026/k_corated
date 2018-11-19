import random
import logging
import argparse
from rs import get_ratings_name_from_dataset, load
import numpy as np
import matplotlib.pyplot as plt
import math


def get_initial_seeds(original_ratings, size, mode):
    if mode == 'random':
        seeds = get_initial_seeds_randomly(original_ratings, size)
    elif mode == 'sort':
        seeds = get_initial_seeds_by_sort(original_ratings, size)
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


def get_best_initial_seeds(original_ratings, size, mode, try_time=10):
    seeds_list = []
    loss_list = []
    for i in range(try_time):
        seeds_list.append(get_initial_seeds(original_ratings, size, mode))
    for seeds in seeds_list:
        logging.info('trying %sth seeds', seeds_list.index(seeds))
        clusters = [Cluster(seed) for seed in seeds]
        k_means_iter_once(clusters)
        loss = loss_of_clusters(clusters)
        loss_list.append(loss)
    min_loss = min(loss_list)
    min_index = loss_list.index(min_loss)
    return seeds_list[min_index]


def get_initial_seeds_by_sort(original_ratings, size):
    from k_corated_by import get_sort_index
    index = get_sort_index(original_ratings)
    slice_size = len(index) // size
    seeds = []
    for i in range(size):
        start = i * slice_size
        if i == size - 1:
            end = len(index) - 1
        else:
            end = start + slice_size - 1
        seed = random.randint(start, end)
        seeds.append(seed)
    return seeds


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
        if len(self.points):
            self.centroid = [i / len(self.points) for i in self.items_sum]
        self._clear()

    def _clear(self):
        self.points.clear()
        self.items_sum = None

    def items_to_keep(self):
        baseline = None
        top = self.top_n_greater_than(baseline)
        zipped = [(self.items_sum[index], index) for index in range(len(self.items_sum))]
        sorted_temp = sorted(zipped, key=lambda x: x[0], reverse=True)
        top_temp = [sorted_temp[i] for i in range(top)]
        top_index = [z[1] for z in top_temp]
        return top_index

    def points_fix(self):
        self.items_sum = self._get_items_sum()

    def _get_items_sum(self):
        temp = [0] * Cluster.original_ratings.shape[1]
        for i in range(Cluster.original_ratings.shape[1]):
            for point in self.points:
                temp[i] += 0 if Cluster.original_ratings[point, i] == 0 else 1
        return temp

    def add_new_point(self, point):
        self.points.append(point)

    def distance_to(self, point):
        point_vec = np.array(normalize(Cluster.original_ratings[point, :]))
        centroid = np.array(self.centroid)
        temp = point_vec - centroid
        return np.sqrt((temp * temp).sum())

    def loss(self):
        s = sum(normalize(self.items_sum)) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(normalize(Cluster.original_ratings[point, :]))
        return s - t

    def lost(self):
        if len(self.points) != 0:
            temp_center = np.array([i / len(self.points) for i in self.items_sum])
            lost = 0
            for point in self.points:
                normalized_point = np.array(normalize(Cluster.original_ratings[point, :]))
                temp = temp_center - normalized_point
                lost += np.sqrt((temp * temp).sum())
            return lost
        else:
            return 0

    def top_n_contribution(self, n):
        temp = self.items_sum
        zipped = [(self.items_sum[index], index) for index in range(len(self.items_sum))]
        sorted_temp = sorted(zipped, key=lambda x: x[0], reverse=True)
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

    def top_n_greater_than(self, baseline):
        temp = self.items_sum
        portion = [i / len(self.points) for i in temp]
        portion.append(baseline)
        portion.sort(reverse=True)
        return portion.index(baseline) - 1

    def contribution_by_baseline(self, baseline):
        n = self.top_n_greater_than(baseline)
        return self.top_n_contribution(n)

    def info(self):
        pass


def contribution_of_clusters(clusters, baseline):
    add = 0
    cost = 0
    for cluster in clusters:
        a, c = cluster.contribution_by_baseline(baseline)
        add += a
        cost += c
    return add, cost


def loss_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.loss()
    return s


def lost_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.lost()
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
        cluster.points_fix()


def update_all(clusters):
    for cluster in clusters:
        cluster.update_centroid()


def analysis_of_clusters(clusters):
    adds = []
    costs = []
    baselines = []
    for i in range(1, 50, 2):
        baseline = i / 100
        add, cost = contribution_of_clusters(clusters, baseline)
        adds.append(add)
        costs.append(cost)
        baselines.append(baseline)
    plt.figure()
    plt.plot(costs, adds)
    plt.savefig('delete-add.jpg')
    plt.figure()
    plt.plot(baselines, [cost / 80000 for cost in costs])
    plt.savefig('baseline-cost.jpg')


def k_means(seeds, threshold=5000):
    clusters = [Cluster(seed) for seed in seeds]
    k_means_iter_once(clusters)
    old_loss = loss_of_clusters(clusters)
    update_all(clusters)
    index = 0
    while True:
        k_means_iter_once(clusters)
        new_loss = loss_of_clusters(clusters)
        diff = old_loss - new_loss
        logging.info('k:%s, iteration:%s, old_loss:%s, new_loss:%s, diff:%s', len(seeds), index, old_loss, new_loss,
                     diff)
        index += 1
        if diff < threshold:
            break
        else:
            old_loss = new_loss
            update_all(clusters)
    return clusters


def find_best_k(original_ratings, mode):
    k_list = [i for i in range(70, 110, 10)]
    loss_list = []
    for k in k_list:
        best_seeds = get_best_initial_seeds(original_ratings, k, mode)
        clusters = k_means(best_seeds)
        loss_list.append(loss_of_clusters(clusters))
    plt.figure()
    plt.plot(k_list, loss_list)
    plt.savefig('k_loss.jpg')
    logging.info(k_list)
    logging.info(loss_list)


def dump_clusters(clusters, k):
    with open('best_clusters_%s.txt' % k, 'w') as file:
        for cluster in clusters:
            file.write('%s' % (cluster.points) + '\n')


def load_clusters(original_ratings, k):
    Cluster.original_ratings = original_ratings
    clusters = []
    with open('best_clusters_%s.txt' % k, 'r') as file:
        lines = file.readlines()
        for line in lines:
            points = eval(line)
            cluster = Cluster(0)
            cluster.points = points
            cluster.points_fix()
            clusters.append(cluster)
    return clusters


def main():
    parser = argparse.ArgumentParser(description='k corating a rating file by a certain web')
    parser.add_argument('-d', '--database', required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('-m', '--mode', required=True)
    parser.add_argument('-a', '--analysis', action='store_true')
    args = parser.parse_args()
    original_ratings = load(get_ratings_name_from_dataset(args.database))
    k = args.k
    mode = args.mode
    need_analysis = args.analysis
    Cluster.original_ratings = original_ratings
    if k != 0:
        best_seeds = get_best_initial_seeds(original_ratings, k, mode)
        best_clusters = k_means(best_seeds)
        dump_clusters(best_clusters, k)
        if need_analysis:
            analysis_of_clusters(best_clusters)
    else:
        find_best_k(original_ratings, mode)


if __name__ == '__main__':
    main()
