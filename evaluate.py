from rs import pd_rating, load, directory, nearest_neighbors_by_fix_number, dataset_choices, get_all_web_files, \
    rating_scale, get_ratings_name_from_dataset
import numpy as np
import math
import argparse
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

original_ratings = None


def adapter(num, type=0):
    def fix(data):
        if data < 1:
            return 1
        if data > 5:
            return 5
        return data

    if type == 'int':
        return fix(int(num))
    elif type == 'round':
        return fix(int(round(num)))
    elif type == 'customize':
        if num <= 1.3:
            return 1
        elif num > 1.3 and num <= 2.6:
            return 2
        elif num > 2.6 and num <= 3.6:
            return 3
        elif num > 3.6 and num <= 4.2:
            return 4
        else:
            return 5
    else:
        raise ValueError


def RMSE(dataset, web, top, adapter_kind):
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
        predicted_rating, des = pd_rating(original_ratings, test_user, test_item, web, nearest_neighbors_by_fix_number,
                                          top)
        predicted_rating = adapter(predicted_rating, adapter_kind)
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
    parser.add_argument('-t', '--top', type=int)
    parser.add_argument('-a', '--adapter')
    args = parser.parse_args()
    data_set = args.dataset
    web_name = args.web
    top = args.top
    adapter_kind = args.adapter
    init(data_set)
    temp_tops = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    if web_name == 'all' and adapter_kind != 'all':
        web_names = get_all_web_files()
        plt.figure()
        plt.xlabel('top')
        plt.ylabel('RMSE')
        for web_name in web_names:
            y = []
            web = load(web_name)
            for temp_top in temp_tops:
                count = RMSE(data_set, web, temp_top, adapter_kind)
                y.append(count['RMSE'])
            plt.plot(temp_tops, y, marker='*', label=web_name)
        plt.legend()
        plt.savefig('RMSE-evaluation-webs.jpg')
    elif web_name != 'all' and adapter_kind == 'all':
        adapter_kinds = ['int', 'round', 'customize']
        plt.figure()
        plt.xlabel('top')
        plt.ylabel('RMSE')
        web = load(web_name)
        for adapter_kind in adapter_kinds:
            y = []
            for temp_top in temp_tops:
                count = RMSE(data_set, web, temp_top, adapter_kind)
                y.append(count['RMSE'])
            plt.plot(temp_tops, y, marker='*', label=adapter_kind)
        plt.legend()
        plt.savefig('RMSE-evaluation-adapters.jpg')
    elif web_name != 'all' and adapter_kind != 'all':
        logging.info(RMSE(data_set, load(web_name), top, adapter_kind))
    else:
        raise ValueError



if __name__ == '__main__':
    main()
