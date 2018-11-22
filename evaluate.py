from rs import pd_rating, load, directory, neareast_neighbors_by_threshold, nearest_neighbors_by_fix_number, supp_item, \
    dataset_choices, get_all_web_files, rating_scale, get_ratings_name_from_dataset
import numpy as np
import math
import argparse
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

original_ratings = None


def RMSE(dataset, web, neighbor_fun, neighbor_para):
    test_set = np.loadtxt(directory + dataset + '.test', delimiter='\t')
    size = test_set.shape[0]
    total = 0
    count_data = {'num': 0, 'ratings': [0] * rating_scale, 'diff': [0] * rating_scale, 'rmse': 0,
                  'diff_2': {'1,3': 0, '2,4': 0, '3,5': 0, '3,1': 0, '4,2': 0, '5,3': 0}}
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
        if predicted_rating == 3 and test_rating == 1:
            count[des]['diff_2']['3,1'] += 1
        elif predicted_rating == 1 and test_rating == 3:
            count[des]['diff_2']['1,3'] += 1
        elif predicted_rating == 2 and test_rating == 4:
            count[des]['diff_2']['2,4'] += 1
        elif predicted_rating == 4 and test_rating == 2:
            count[des]['diff_2']['4,2'] += 1
        elif predicted_rating == 3 and test_rating == 5:
            count[des]['diff_2']['3,5'] += 1
        elif predicted_rating == 5 and test_rating == 3:
            count[des]['diff_2']['5,3'] += 1
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
            return count
        elif threshold and top is None:
            count = RMSE(data_set, web, neareast_neighbors_by_threshold, threshold)
            return count

    if web_name != 'all':
        count = rmse(web_name)
        logging.info(count)
    else:
        web_names = get_all_web_files(suffix)
        temp_tops = [5, 10, 15, 20, 25, 40, 50, 60]
        plt.figure()
        plt.xlabel('top')
        plt.ylabel('RMSE')
        for web_name in web_names:
            y = []
            for temp_top in temp_tops:
                global top
                top = temp_top
                count = rmse(web_name)
                y.append(count['RMSE'])
            plt.plot(temp_tops, y, marker='*', label=web_name)
        plt.legend()
        plt.savefig('RMSE-evaluation.jpg')


if __name__ == '__main__':
    main()
