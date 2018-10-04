from rs import supp_user, user_size, item_size, pd_rating, default_neighbors,dump
from functools import cmp_to_key
import numpy as np
import logging
import pandas as pd
import argparse

default_predict_fun = pd_rating


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
    ratings = np.insert(ratings, item_size, [i + 1 for i in range(user_size)], 1)
    temp = [ratings[i, :] for i in range(user_size)]
    temp.sort(key=cmp_to_key(mycmp))
    ret = np.zeros(shape=(user_size, item_size + 1))
    for i in range(user_size):
        ret[i, :] = temp[i].copy()
    return ret


def part_k_corated(ratings, k):
    candi = []
    temp = {}
    for i in range(user_size):
        size = len(supp_user(ratings[i, :]))
        if size not in temp.keys():
            temp[size] = [i, i]
        else:
            temp[size][1] = i
    for value in temp.values():
        if value[1] - value[0] + 1 < k:
            continue
        else:
            while value[1] - value[0] + 1 >= k:
                right = are_corated(ratings, value[0], value[0] + k - 1)
                if not right[0]:
                    value[0] = right[1]
                else:
                    j = right[1]
                    while j <= value[1]:
                        if is_corated(ratings[value[0], :], ratings[j, :]):
                            j += 1
                        else:
                            break
                    candi.extend([i for i in range(value[0], j)])
                    value[0] = j
    if candi:
        return ratings[candi].copy(), np.delete(ratings, candi, 0)
    else:
        return None, ratings.copy()


def is_corated(user1, user2):
    if supp_user(user1) == supp_user(user2):
        return True
    return False


def are_corated(ratings, start, end):
    for i in range(start, end):
        if not is_corated(ratings[i, :], ratings[i + 1, :]):
            return False, i + 1
    return True, end + 1


def k_corating(k, non_k_matrix, original_ratings, neighbor_ids, web):
    remain = non_k_matrix.shape[0]
    start = 0
    while remain > 0:
        if remain >= k:
            temp_range = [start, start + k]
            while temp_range[1] < non_k_matrix.shape[0]:
                if is_corated(non_k_matrix[temp_range[1] - 1, :], non_k_matrix[temp_range[1], :]):
                    temp_range[1] += 1
                else:
                    break
            if non_k_matrix.shape[0] - temp_range[1] < k:
                temp_range[1] = non_k_matrix.shape[0]
        else:
            temp_range = [start, non_k_matrix.shape[0]]
        logging.info('trying to fill lines in [%s,%s]', temp_range[0], temp_range[1] - 1)
        items_need_to_rate = set()
        for i in range(temp_range[0], temp_range[1]):
            items_need_to_rate = items_need_to_rate.union(supp_user(non_k_matrix[i, :]))
        items_need_to_rate = list(items_need_to_rate)
        for item_id in items_need_to_rate:
            for i in range(temp_range[0], temp_range[1]):
                if non_k_matrix[i][item_id] == 0:
                    non_k_matrix[i][item_id] = pd_rating(original_ratings, int(non_k_matrix[i][-1]) - 1, item_id,
                                                         neighbor_ids, web)
        start = temp_range[1]
        remain -= temp_range[1] - temp_range[0]


def k_corate(k, ratings_name, web_name, neighbor_ids=default_neighbors):
    ratings = np.loadtxt(ratings_name, delimiter=',')
    web = np.loadtxt(web_name, delimiter=',')
    logging.info('sorting the ratings:%s', ratings_name)
    sorted_ratings = sort(ratings)
    logging.info('dividing the sorted ratings from %s', ratings_name)
    k_coreted_part, non_k_corated_part = part_k_corated(sorted_ratings, k)
    logging.info('%s corating the non-%s-corated part by web:%s', k, k, web_name)
    k_corating(k, non_k_corated_part, ratings, neighbor_ids, web)
    logging.info('combing two parts into one')
    if k_coreted_part is not None:
        ret = np.insert(k_coreted_part, k_coreted_part.shape[0], non_k_corated_part, 0)
    else:
        ret = non_k_corated_part
    ret_no_index = np.delete(ret, ret.shape[1] - 1, 1)
    return ret_no_index, ret


def main():
    parser=argparse.ArgumentParser(description='k corating a rating file by a certain web')
    parser.add_argument('-r','--ratings',required=True)
    parser.add_argument('-k',required=True,type=int)
    parser.add_argument('-w','--web',required=True)
    args=parser.parse_args()
    k=args.k
    web_name=args.web
    ratings_name=args.file
    with_index,without_index=k_corate(k,ratings_name,web_name)
    filename='%s_corated_ratings_from_%s_by_%s_with_index' % (k,ratings_name[:7],web_name[8:-4])
    dump(filename,with_index)
    filename = '%s_corated_ratings_from_%s_by_%s_without_index' % (k, ratings_name[:7], web_name[8:-4])
    dump(filename,without_index)


if __name__ == '__main__':
    main()
