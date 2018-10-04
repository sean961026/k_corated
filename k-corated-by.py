from rs import supp_user, users, items, pd_rating
from functools import cmp_to_key
import numpy as np
import logging
import pandas as pd
import sys

input_k = int(sys.argv[1])
input_rating_file = sys.argv[2]
input_mode = sys.argv[3]
default_predict_fun=pd_rating
default_neighbors=[i for i in range(len(users))]

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
    logging.info('sorting the ratings')
    ratings = np.insert(ratings, len(items), [i + 1 for i in range(len(users))], 1)
    temp = [ratings[i, :] for i in range(len(users))]
    temp.sort(key=cmp_to_key(mycmp))
    ret = np.zeros(shape=(len(users), len(items) + 1))
    for i in range(len(users)):
        ret[i, :] = temp[i].copy()
    return ret


def part_k_corated(ratings, k):
    logging.info('dividing the ratings into two')
    candi = []
    temp = {}
    for i in range(len(users)):
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


def k_corating(k, non_k_matrix, predict_fun, **kwargs):
    logging.info('filling the non_k_corated matrix which is shape(%s,%s), k is %s',
                 non_k_matrix.shape[0], non_k_matrix.shape[1], k)
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
                    # ratings, mode = 'default', trust_web = None
                    kwargs.update({'user_id': int(non_k_matrix[i][-1] - 1), 'item_id': item_id})
                    if 'neighbor_ids' not in kwargs:
                        kwargs.update({'neighbor_ids':default_neighbors})
                    non_k_matrix[i][item_id] = predict_fun(**kwargs)
        start = temp_range[1]
        remain -= temp_ra1nge[1] - temp_range[0]


def k_corate(k, ratings, predict_fun, **kwargs):
    sorted_ratings = sort(ratings)
    k_coreted_part, non_k_corated_part = part_k_corated(sorted_ratings, k)
    k_corating(k, non_k_corated_part, predict_fun, **kwargs)
    logging.info('combining the two part')
    if k_coreted_part is not None:
        ret = np.insert(k_coreted_part, k_coreted_part.shape[0], non_k_corated_part, 0)
    else:
        ret = non_k_corated_part
    ret_no_index = np.delete(ret, ret.shape[1] - 1, 1)
    return ret_no_index, ret


def main(k, rating_file, mode):
    original_ratings = np.loadtxt(rating_file + '_ratings.csv', delimiter=',')
    trust_web = np.loadtxt(rating_file + '_trust_web.csv', delimiter=',')
    k_corated_ratings, k_corated_ratings_with_reference = k_corate(k, original_ratings, default_predict_fun,
                                                                   **{'mode': mode, 'trust_web': trust_web})
    pd.DataFrame(k_corated_ratings).to_csv(rating_file + '_' + str(k) + '_corated_ratings_by_' + mode + '.csv',
                                           index=False,header=False)
    logging.info('dump %s to csv', rating_file + '_' + str(k) + '_corated_ratings_by_'+mode)
    pd.DataFrame(k_corated_ratings_with_reference).to_csv(
        rating_file + '_' + str(k) + '_corated_ratings_with_index_by_'+mode+'.csv', index=False, header=False)
    logging.info('dump %s to csv', rating_file + '_' + str(k) + '_corated_ratings_with_index_by_' + mode)


if __name__ == '__main__':
    main(input_k, input_rating_file, input_mode)
