from rs import pd_rating, directory, user_size
import numpy as np
import math
import argparse
import logging

dataset_choices = ['u1', 'u2', 'u3', 'u4', 'u5']
default_neighbors = [i for i in range(user_size)]


def RMSE(test_set_file, base_ratings, neighbor_ids, web):
    original_ratings = np.loadtxt(base_ratings, delimiter=',')
    test_set = np.loadtxt(directory + test_set_file, delimiter='\t')
    size = test_set.shape[0]
    total = 0
    for i in range(size):
        record = test_set[i, :]
        test_user = int(record[0] - 1)
        test_item = int(record[1] - 1)
        test_rating = record[2]
        predicted_rating = pd_rating(original_ratings, test_user, test_item, neighbor_ids, web)
        total += (test_rating - predicted_rating) ** 2
    return math.sqrt(total / size)


def main():
    parser = argparse.ArgumentParser(description='RMSE test for a certain dataset')
    parser.add_argument('-d', '--dataset', required=True, choices=dataset_choices)
    parser.add_argument('-w', '--web', required=True)
    args = parser.parse_args()
    data_set = args.dataset
    web_name = args.web
    original_ratings = np.loadtxt(data_set + '.base_ratings.csv', delimiter=',')
    web = np.loadtxt(web_name, delimiter=',')
    test_set_file = data_set + '.test'
    ans = RMSE(test_set_file, original_ratings, default_neighbors, web)
    logging.info('the RMSE result of %s predicted by %s is %s', data_set, web_name[:-4], ans)


if __name__ == '__main__':
    main()
