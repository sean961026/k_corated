from rs import pd_rating, directory, user_size
import numpy as np
import math
import argparse
import logging

dataset_choices = ['u1', 'u2', 'u3', 'u4', 'u5']
default_neighbors = [i for i in range(user_size)]


def RMSE(test_set_file_name, base_ratings_file_name, neighbor_ids, web_name):
    original_ratings = np.loadtxt(base_ratings_file_name, delimiter=',')
    test_set = np.loadtxt(test_set_file_name, delimiter='\t')
    web=np.loadtxt(web_name,delimiter=',')
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
    test_set_file_name = directory+data_set + '.test'
    ans = RMSE(test_set_file_name, data_set + '_ratings.csv', default_neighbors, web_name)
    logging.info('the RMSE result of %s predicted by %s is %s', data_set, web_name[:-4], ans)


if __name__ == '__main__':
    main()
