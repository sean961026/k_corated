from rs import pd_rating, load, directory, neareast_neighbors_by_threshold, nearest_neighbors_by_fix_number, supp_item, \
    dataset_choices, get_all_web_files, rating_scale, get_ratings_name_from_dataset
import numpy as np
import math
import argparse
import logging
from copy import deepcopy

original_ratings = None


def RMSE(dataset, web, neighbor_fun, neighbor_para):
    test_set = np.loadtxt(directory + dataset + '.test', delimiter='\t')
    size = test_set.shape[0]
    total = 0
    count_data = {'num': 0, 'ratings': [0] * rating_scale, 'diff': [0] * rating_scale, 'rmse': 0}
    count = {'normal': deepcopy(count_data), 'over': deepcopy(count_data), 'below': deepcopy(count_data),
             'exception': deepcopy(count_data)}
    for i in range(size):
        record = test_set[i, :]
        test_user = int(record[0] - 1)
        test_item = int(record[1] - 1)
        test_rating = record[2]
        predicted_rating, des = pd_rating(original_ratings, test_user, test_item, web, neighbor_fun, neighbor_para)
        count[des]['num'] += 1
        count[des]['ratings'][predicted_rating - 1] += 1
        count[des]['diff'][abs(int(predicted_rating - test_rating))] += 1
        temp = (test_rating - predicted_rating) ** 2
        count[des]['rmse'] += temp
        total += temp
    count['RMSE'] = math.sqrt(total / size)
    for des in ['normal', 'over', 'below', 'exception']:
        count[des]['rmse'] = math.sqrt(count[des]['rmse'] / count[des]['num'])
    return count


def init(data_set):
    global original_ratings
    original_ratings = load(get_ratings_name_from_dataset(data_set))


def main():
    parser = argparse.ArgumentParser(description='RMSE test for a certain dataset')
    parser.add_argument('-d', '--dataset', required=True, choices=dataset_choices)
    parser.add_argument('-w', '--web', required=True)
    parser.add_argument('-s', '--suffix')
    parser.add_argument('-t', '--threshold')
    parser.add_argument('--top', type=int)
    args = parser.parse_args()
    data_set = args.dataset
    web_name = args.web
    top = args.top
    threshold = args.threshold
    suffix = args.suffix
    init(data_set)

    def rmse(webname):
        web = load(webname)
        if top and threshold is None:
            count = RMSE(data_set, web, nearest_neighbors_by_fix_number, top)
            logging.info('count:%s ', count)
        elif threshold and top is None:
            count = RMSE(data_set, web, neareast_neighbors_by_threshold, threshold)
            logging.info('count:%s ', count)

    if web_name != 'all':
        rmse(web_name)
    else:
        web_names = get_all_web_files(suffix)
        for web_name in web_names:
            rmse(web_name)


if __name__ == '__main__':
    main()
