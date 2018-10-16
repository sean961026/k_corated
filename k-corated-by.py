from rs import supp_user, pd_rating, dump, load, unknown_rating, nearest_neighbors_by_fix_number, \
    neareast_neighbors_by_threshold, dataset_choices, get_ratings_name_from_dataset, get_all_web_files
from functools import cmp_to_key
import numpy as np
import logging
import argparse

original_ratings = None
sorted_ratings = None
neighbor_fun = None
neighbor_para = None
web = None

def set_cmp(user1, user2):
    items_1 = supp_user(user1)
    items_2 = supp_user(user2)
    while len(items_1) != 0:
        if min(items_1) < min(items_2):
            return -1
        elif min(items_1) > min(items_2):
            return 1
        else:
            min_value = min(items_1)
            items_1.remove(min_value)
            items_2.remove(min_value)
    return 0


def mycmp(user1, user2):
    if len(supp_user(user1)) < len(supp_user(user2)):
        return -1
    elif len(supp_user(user1)) > len(supp_user(user2)):
        return 1
    else:
        return set_cmp(user1, user2)


def sort(ratings):
    user_size = ratings.shape[0]
    item_size = ratings.shape[1]
    ratings = np.insert(ratings, item_size, [i + 1 for i in range(user_size)], 1)
    temp = [ratings[i, :] for i in range(user_size)]
    temp.sort(key=cmp_to_key(mycmp))
    ret = np.zeros(shape=(user_size, item_size + 1))
    for i in range(user_size):
        ret[i, :] = temp[i].copy()
    return ret


def k_corating_slice(sorted_ratings, myslice):  # [start,end)
    logging.info('k corating slice(%s,%s)', myslice.start, myslice.stop)
    part_ratings = sorted_ratings[myslice, :]
    items_need_to_rate = set()
    for record in part_ratings:
        items_need_to_rate = items_need_to_rate.union(supp_user(record))
    for item_id in items_need_to_rate:
        for record in part_ratings:
            if record[item_id] == unknown_rating:
                record[item_id] = round(
                    pd_rating(original_ratings, int(record[-1] - 1), item_id, web, neighbor_fun, neighbor_para))


def k_corating_all(sorted_ratings, k):
    start = 0
    while sorted_ratings.shape[0] - start >= k:
        myslice = slice(start, start + k)
        k_corating_slice(sorted_ratings, myslice)
        start += k
    myslice = slice(start, sorted_ratings.shape[0])
    k_corating_slice(sorted_ratings, myslice)
    index_translator = sorted_ratings[:, -1]
    k_corated = np.delete(sorted_ratings, sorted_ratings.shape[1] - 1, 1)
    return k_corated, index_translator


def init(dataset, threshold, top):
    global original_ratings, neighbor_fun, neighbor_para, sorted_ratings
    original_ratings = load(get_ratings_name_from_dataset(dataset))
    sorted_ratings = sort(original_ratings)
    if threshold and top is None:
        neighbor_fun = neareast_neighbors_by_threshold
        neighbor_para = threshold
    elif top and threshold is None:
        neighbor_fun = nearest_neighbors_by_fix_number
        neighbor_para = top
    else:
        raise ValueError


def get_k_corated_name_by_atrr(dataset, k, web_name, threshold, top):
    web_name = web_name[4:-4]
    if threshold and top is None:
        name = 'k_ratings_%s_%s_%s_th%s.csv' % (k, dataset, web_name, str(threshold))
    elif top and threshold is None:
        name = 'k_ratings_%s_%s_%s_top%s.csv' % (k, dataset, web_name, str(top))
    else:
        raise ValueError
    return name


def get_k_corated_index_by_att(dataset, k, web_name, threshold, top):
    web_name = web_name[4:-4]
    if threshold and top is None:
        name = 'index_%s_%s_%s_th%s.csv' % (k, dataset, web_name, str(threshold))
    elif top and threshold is None:
        name = 'index_%s_%s_%s_top%s.csv' % (k, dataset, web_name, str(top))
    else:
        raise ValueError
    return name


def main():
    parser = argparse.ArgumentParser(description='k corating a rating file by a certain web')
    parser.add_argument('-d', '--dataset', required=True, choices=dataset_choices)
    parser.add_argument('-w', '--web', required=True)
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('-s', '--suffix')
    parser.add_argument('-t', '--threshold')
    parser.add_argument('--top', type=int)
    args = parser.parse_args()
    data_set = args.dataset
    web_name = args.web
    top = args.top
    threshold = args.threshold
    k = args.k
    suffix = args.suffix
    init(data_set, threshold, top)

    def k_corated(webname):
        global web
        web = load(webname)
        k_corated, index_translator = k_corating_all(sorted_ratings.copy(), k)
        k_file_name = get_k_corated_name_by_atrr(data_set, k, webname, threshold, top)
        index_file_name = get_k_corated_index_by_att(data_set, k, webname, threshold, top)
        dump(k_file_name, k_corated)
        dump(index_file_name, index_translator)

    if web_name != 'all':
        k_corated(web_name)
    else:
        web_names = get_all_web_files(suffix)
        for web_name in web_names:
            k_corated(web_name)


if __name__ == '__main__':
    main()
