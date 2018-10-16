from rs import pd_rating, load, directory, neareast_neighbors_by_threshold, nearest_neighbors_by_fix_number, supp_item, \
    dataset_choices, get_all_web_files, rating_scale
import numpy as np
import math
import argparse
import logging


def RMSE(dataset, web, neighbor_fun, neighbor_para):
    test_set = np.loadtxt(directory + dataset + '.test', delimiter='\t')
    original_ratings = load(dataset + '.base.csv')
    size = test_set.shape[0]
    total = 0
    diff = [0] * rating_scale
    count = {'normal': diff.copy(), 'over': diff.copy(), 'below': diff.copy(), 'exception': diff.copy()}
    for i in range(size):
        record = test_set[i, :]
        test_user = int(record[0] - 1)
        test_item = int(record[1] - 1)
        test_rating = record[2]
        candidates = supp_item(original_ratings[:, test_item])
        neighbors = neighbor_fun(candidates, test_user, web, neighbor_para)
        predicted_rating, des = pd_rating(original_ratings, test_user, test_item, web, neighbors)
        count[des][abs(int(predicted_rating - test_rating))] += 1
        total += (test_rating - predicted_rating) ** 2
    count['RMSE'] = math.sqrt(total / size)
    return count


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

    def rmse(webname):
        web = load(webname)
        if top and threshold is None:
            count = RMSE(data_set, web, nearest_neighbors_by_fix_number, top)
            logging.info('the RMSE result of %s predicted by %s is %s ', data_set, web_name[:-4], count)
        elif threshold and top is None:
            count = RMSE(data_set, web, neareast_neighbors_by_threshold, threshold)
            logging.info('the RMSE result of %s predicted by %s is %s', data_set, web_name[:-4], count)

    if web_name != 'all':
        rmse(web_name)
    else:
        web_names = get_all_web_files(suffix)
        for web_name in web_names:
            rmse(web_name)


if __name__ == '__main__':
    main()
