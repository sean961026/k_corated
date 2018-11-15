import random
import numpy as np
import logging
import argparse
from rs import get_ratings_name_from_dataset, load
import numpy as np


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
        self.start = sum(self.centroid)

    def update_centroid(self):
        if len(self.points):
            temp = self._get_items_sum()
            self.centroid = [i / len(self.points) for i in temp]

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
        dis = 0
        for i in range(len(point_vec)):
            dis += (point_vec[i] - self.centroid[i]) ** 2
        return dis

    def dis_sum(self):
        s = sum(self._get_corated()) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(normalize(Cluster.original_ratings[point, :]))
        return s - t

    def items_to_keep(self):
        # n = sum(self.centroid)
        n = self.start
        temp = self._get_items_sum()
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])
        top_temp = [sorted_temp[i] for i in range(n)]
        top_index = [z[1] for z in top_temp]
        return top_index

    def top_n_contribution(self, n):
        temp = self._get_items_sum()
        zip_temp = [(temp[i], i) for i in range(len(temp))]
        sorted_temp = sorted(zip_temp, reverse=True, key=lambda x: x[0])
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

    def copy(self):
        ins = Cluster(0)
        ins.centroid = self.centroid
        ins.points = self.points.copy()
        return ins

    def info(self):
        temp = self._get_items_sum()
        size = len(temp)
        portion = [i / len(self.points) for i in temp]
        portion.sort(reverse=True)

        def get_top_by_portion_base(base):
            if portion[0] < base:
                return 0
            for i in range(size):
                if portion[i] < base:
                    return i - 1
            return size

        for top in [get_top_by_portion_base(0.1), get_top_by_portion_base(0.2), get_top_by_portion_base(0.3)]:
            p = portion[top]
            add, cost = self.top_n_contribution(top)
            logging.info({'top': top, 'portion': p, 'add': add, 'cost': cost})


def dis_of_clusters(clusters):
    # add = add_of_clusters(clusters)
    # delete = delete_of_clusters(clusters)
    # return add + 10 * delete
    dis = 0
    for cluster in clusters:
        dis += cluster.dis_sum()
    return dis


def delete_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.cost()
    return s


def add_of_clusters(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.new_dis()
    return s


def k_means_iter_once(clusters):
    point_size = Cluster.original_ratings.shape[0]
    for point in range(point_size):
        dis = []
        for cluster in clusters:
            dis.append(cluster.distance_to(point))
        min_cluster = dis.index(min(dis))
        clusters[min_cluster].add_new_point(point)


def update_all(clusters):
    for cluster in clusters:
        cluster.update_centroid()


def clear_all(clusters):
    for cluster in clusters:
        cluster.clear()


def copy_all(clusters):
    cp = []
    for cluster in clusters:
        single_cp = cluster.copy()
        cp.append(single_cp)
    return cp


def k_means(original_ratings, k, mode):
    Cluster.original_ratings = original_ratings
    seeds = get_initial_seeds(original_ratings, k, mode)
    clusters = [Cluster(seed) for seed in seeds]
    for i in range(10):
        k_means_iter_once(clusters)
        for cluster in clusters:
            cluster.info()
        update_all(clusters)
        clear_all(clusters)
    return clusters


def dump_clusters(clusters, k):
    with open('best_clusters_%s.txt' % k, 'w') as file:
        for cluster in clusters:
            file.write('%d:%s' % (sum(cluster.centroid), cluster.points) + '\n')


def load_clusters(original_ratings, k):
    Cluster.original_ratings = original_ratings
    clusters = []
    with open('best_clusters_%s.txt' % k, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cen_sum, points = line.split(':')
            cen_sum = int(cen_sum)
            points = eval(points)
            cluster = Cluster(0)
            cluster.centroid = [cen_sum]
            cluster.points = points
            clusters.append(cluster)
    return clusters


def main():
    parser = argparse.ArgumentParser(description='k corating a rating file by a certain web')
    parser.add_argument('-d', '--database', required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('-m', '--mode', required=True)
    args = parser.parse_args()
    best_clusters = k_means(load(get_ratings_name_from_dataset(args.database)), args.k, args.mode)
    dump_clusters(best_clusters, args.k)


if __name__ == '__main__':
    main()
