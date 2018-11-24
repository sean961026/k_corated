import random
import logging
import argparse
from rs import get_ratings_name_from_dataset, load, dump
import numpy as np
import matplotlib.pyplot as plt
import os


def get_initial_seeds(original_ratings, normalized_ratings, size, mode, dis_map):
    if mode == 'random':
        seeds = get_initial_seeds_randomly(normalized_ratings, size)
    elif mode == 'rsort':
        seeds = get_initial_seeds_by_rsort(original_ratings, size)
    elif mode == 'dsort':
        seeds = get_initial_seeds_by_dsort(original_ratings, size)
    elif mode == 'density':
        seeds = get_initial_seeds_by_density(normalized_ratings, size, dis_map)
    else:
        raise ValueError
    return seeds


def get_initial_seeds_by_density(normalized_ratings, size, dis_map=None):
    user_size = normalized_ratings.shape[0]
    seeds = []

    def dis(i, j):
        norm_i = normalized_ratings[i, :]
        norm_j = normalized_ratings[j, :]
        np_i = np.array(norm_i)
        np_j = np.array(norm_j)
        temp = np_j - np_i
        return np.sqrt((temp * temp).sum())

    if dis_map is None:
        dis_map = np.zeros(shape=(user_size, user_size))
        for i in range(user_size):
            logging.info('filling %dth row', i)
            for j in range(user_size):
                dis_map[i, j] = dis(i, j)
        dump('dis_map.csv', dis_map)

    def RS(i):
        return dis_map[i, :].sum()

    def get_SRS():
        srs = []
        for i in range(user_size):
            srs.append((RS(i), i))
        srs.sort(key=lambda x: x[0])
        return srs

    SRS = get_SRS()
    seeds.append(SRS[0][1])

    def find_l_th_seed(l):
        S = []
        for i in range(user_size):
            if i not in seeds:
                g = []
                for j in range(l - 1):
                    g.append(dis_map[i, seeds[j]])
                S.append((min(g), i))
        temp = [x[0] for x in S]
        compensation = [(max(temp) + 1, index) for index in seeds]
        S.extend(compensation)
        S.sort(key=lambda x: x[0])
        SDV = S
        alfa = 0.1
        NDDI = []
        index_SRS = [x[1] for x in SRS]
        index_SDV = [x[1] for x in SDV]
        for i in range(user_size):
            if i not in seeds:
                weighted_rank = alfa * index_SDV.index(i) + (1 - alfa) * index_SRS.index(i)
                NDDI.append((weighted_rank, i))
        NDDI.sort(key=lambda x: x[0])
        return NDDI[0][1]

    for i in range(2, size + 1):
        seeds.append(find_l_th_seed(i))
    return seeds


def get_initial_seeds_by_dsort(original_ratings, size):
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
        seed = (start + end) // 2
        seeds.append(seed)
    return seeds


def get_initial_seeds_randomly(normalized_ratings, random_size):
    user_size = normalized_ratings.shape[0]
    if user_size < random_size:
        raise ValueError
    else:
        all = [i for i in range(user_size)]
        return random.sample(all, random_size)


def get_best_initial_seeds(original_ratings, nomalized_ratings, size, mode, dis_map, try_time=5):
    if mode in ['dsort', 'density']:
        return get_initial_seeds(original_ratings, nomalized_ratings, size, mode, dis_map)
    seeds_list = []
    loss_list = []
    for i in range(try_time):
        seeds_list.append(get_initial_seeds(original_ratings, nomalized_ratings, size, mode, dis_map))
    for seeds in seeds_list:
        logging.info('trying %sth seeds', seeds_list.index(seeds))
        clusters = [Cluster(seed) for seed in seeds]
        k_means_iter_once(clusters)
        loss = loss_of_clusters(clusters)
        loss_list.append(loss)
    min_loss = min(loss_list)
    min_index = loss_list.index(min_loss)
    return seeds_list[min_index]


def get_initial_seeds_by_rsort(original_ratings, size):
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
    normalized_ratings = None

    def __init__(self, centroid_index):
        self.centroid = Cluster.normalized_ratings[centroid_index, :]
        self.points = []

    def update_centroid(self):
        if len(self.points):
            self.centroid = [i / len(self.points) for i in self.items_sum]
        self._clear()

    def _clear(self):
        self.points.clear()
        self.items_sum = None

    def items_to_keep(self):
        baseline = 0.05
        top = self.top_n_greater_than(baseline)
        zipped = [(self.items_sum[index], index) for index in range(len(self.items_sum))]
        sorted_temp = sorted(zipped, key=lambda x: x[0], reverse=True)
        top_temp = [sorted_temp[i] for i in range(top)]
        top_index = [z[1] for z in top_temp]
        return top_index

    def points_fix(self):
        self.items_sum = self._get_items_sum()

    def _get_items_sum(self):
        temp = [0] * Cluster.normalized_ratings.shape[1]
        for i in range(Cluster.normalized_ratings.shape[1]):
            for point in self.points:
                temp[i] += Cluster.normalized_ratings[point, i]
        return temp

    def add_new_point(self, point):
        self.points.append(point)

    def distance_to(self, point):
        point_vec = np.array(Cluster.normalized_ratings[point, :])
        centroid = np.array(self.centroid)
        temp = point_vec - centroid
        return np.sqrt((temp * temp).sum())

    def loss(self):
        s = sum(normalize(self.items_sum)) * len(self.points)
        t = 0
        for point in self.points:
            t += sum(Cluster.normalized_ratings[point, :])
        return s - t

    def lost(self):
        if len(self.points) != 0:
            temp_center = np.array([i / len(self.points) for i in self.items_sum])
            lost = 0
            for point in self.points:
                normalized_point = np.array(Cluster.normalized_ratings[point, :])
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
    point_size = Cluster.normalized_ratings.shape[0]
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


def find_best_k(original_ratings, normalized_ratings, modes, dis_map):
    plt.figure()
    plt.xlabel('K')
    plt.ylabel('Numbers To Be Added')
    k_list = [10, 30, 50, 75, 100, 125, 150, 200, 250, 300]
    density_data = []
    random_data = []
    for mode in modes:
        loss_list = []
        for k in k_list:
            best_seeds = get_best_initial_seeds(original_ratings, normalized_ratings, k, mode, dis_map)
            clusters = k_means(best_seeds)
            loss_list.append(loss_of_clusters(clusters))
        if mode == 'density':
            density_data = loss_list.copy()
        elif mode == 'random':
            random_data = loss_list.copy()
        logging.info('%s:%s', mode, loss_list)
        plt.plot(k_list, loss_list, marker='*', label=mode)
    plt.legend()
    if len(modes) != 1:
        plt.savefig('k_add_all.jpg')
        plt.figure()
        plt.xlabel('K')
        plt.ylabel('Numbers To Be Added')
        plt.plot(k_list, density_data, marker='*', label='density')
        plt.plot(k_list, random_data, marker='*', label='random')
        plt.legend()
        plt.savefig('k_add.jpg')
    else:
        plt.savefig('k_add_%s.jpg' % modes[0])


def dump_clusters(clusters):
    k = len(clusters)
    with open('best_clusters_%s.txt' % k, 'w') as file:
        for cluster in clusters:
            file.write('%s' % (cluster.points) + '\n')


def load_clusters(normalized_ratings, k):
    Cluster.normalized_ratings = normalized_ratings
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
    if os.path.exists('nratings_' + args.database + '.csv'):
        normalized_ratings = load('nratings_' + args.database)
    else:
        normalized_ratings = np.zeros(shape=(original_ratings.shape), dtype=int)
        for i in range(original_ratings.shape[0]):
            normalized_ratings[i, :] = normalize(original_ratings[i, :])
        dump('nratings_%s.csv' % args.database, normalized_ratings)
    Cluster.normalized_ratings = normalized_ratings
    if os.path.exists('dis_map.csv'):
        dis_map = load('dis_map.csv')
    else:
        dis_map = None
    if k != 0:
        best_seeds = get_best_initial_seeds(original_ratings, normalized_ratings, k, mode, dis_map)
        best_clusters = k_means(best_seeds)
        dump_clusters(best_clusters)
        if need_analysis:
            analysis_of_clusters(best_clusters)
    else:
        if mode == 'all':
            modes = ['density', 'dsort', 'rsort', 'random']
        else:
            modes = [mode]
        find_best_k(original_ratings, normalized_ratings, modes, dis_map)


if __name__ == '__main__':
    main()
