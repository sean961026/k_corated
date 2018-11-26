from rs import pd_rating, load, directory, dataset_choices, get_all_web_files, rating_scale, \
    get_ratings_name_from_dataset, unknown_rating
import numpy as np
import math
import argparse
import logging
from copy import deepcopy
import matplotlib.pyplot as plt


def adapter(num, type='round'):
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
    elif type == 'int1':
        return fix(int(num) + 1)
    elif type == 'customize':
        if num <= 1.3:
            return 1
        elif 1.3 < num <= 2.6:
            return 2
        elif 2.6 < num <= 3.6:
            return 3
        elif 3.6 < num <= 4.2:
            return 4
        else:
            return 5
    else:
        raise ValueError


def RMSE(test_set, original_ratings, web, top, adapter_kind):
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
        if original_ratings[test_user, test_item] == unknown_rating:
            predicted_rating, des = pd_rating(original_ratings, test_user, test_item, web, top)
            predicted_rating = adapter(predicted_rating, adapter_kind)
        else:
            predicted_rating = int(original_ratings[test_user, test_item])
            des = 'normal'
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
    temp_tops = [5, 10, 20, 40, 60, 70, 80, 90, 100]
    original_ratings = load(get_ratings_name_from_dataset(data_set))
    test_set = np.loadtxt(directory + data_set + '.test', delimiter='\t')
    if web_name == 'all' and adapter_kind != 'all':
        web_names = get_all_web_files()
        plt.figure()
        plt.xlabel('top')
        plt.ylabel('RMSE')
        best = {'top': None, 'RMSE': 2, 'web': None}
        for web_name in web_names:
            y = []
            web = load(web_name)
            for temp_top in temp_tops:
                logging.info({'web': web_name, 'top': temp_top, 'adapter': adapter_kind})
                count = RMSE(test_set, original_ratings, web, temp_top, adapter_kind)
                y.append(count['RMSE'])
            if min(y) < best['RMSE']:
                best = {'web': web_name, 'RMSE': min(y), 'top': temp_tops[y.index(min(y))]}
            logging.info('%s:%s', web_name, y)
            plt.plot(temp_tops, y, marker='*', label=web_name)
        plt.legend()
        plt.savefig('top-RMSE-webs.jpg')
        logging.info('best:%s', best)
    elif web_name != 'all' and adapter_kind == 'all':
        adapter_kinds = ['int', 'round', 'customize', 'int1']
        plt.figure()
        plt.xlabel('top')
        plt.ylabel('RMSE')
        web = load(web_name)
        best = {'top': None, 'RMSE': 2, 'adapter': None}
        for adapter_kind in adapter_kinds:
            y = []
            for temp_top in temp_tops:
                logging.info({'web': web_name, 'top': temp_top, 'adapter': adapter_kind})
                count = RMSE(test_set, original_ratings, web, temp_top, adapter_kind)
                y.append(count['RMSE'])
            if min(y) < best['RMSE']:
                best = {'adapter': adapter_kind, 'RMSE': min(y), 'top': temp_tops[y.index(min(y))]}
            logging.info('%s:%s', adapter_kind, y)
            plt.plot(temp_tops, y, marker='*', label=adapter_kind)
        plt.legend()
        plt.savefig('top-RMSE-adapters.jpg')
        logging.info('best:%s', best)
    elif web_name != 'all' and adapter_kind != 'all':
        logging.info(RMSE(test_set, original_ratings, load(web_name), top, adapter_kind))
    else:
        raise ValueError


if __name__ == '__main__':
    main()
