from rs import supp_user, users, items, original_ratings, pd_rating
from functools import cmp_to_key
import numpy as np
import logging
import pandas as pd
def set_cmp(user1, user2):
    items_1 = supp_user(user1)
    items_2 = supp_user(user2)
    while len(items_1)!=0:
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
    ratings = np.insert(ratings, len(items), [i + 1 for i in range(len(users))], 1)
    temp = [ratings[i, :] for i in range(len(users))]
    temp.sort(key=cmp_to_key(mycmp))
    ret = np.zeros(shape=(len(users), len(items) + 1))
    for i in range(len(users)):
        ret[i, :] = temp[i].copy()
    return ret


def part_k_corated(ratings, k):
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
                right = are_corated(ratings, value[0], value[0] + k-1)
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


def k_corating(k, non_k_matrix, trust_web):
    remain = non_k_matrix.shape[0]
    start = 0
    while remain > 0:
        if remain >= k:
            temp_range = [start, start + k]
            while temp_range[1]<non_k_matrix.shape[0]:
                if is_corated(non_k_matrix[temp_range[1] - 1, :], non_k_matrix[temp_range[1], :]):
                    temp_range[1] += 1
                else:
                    break
            if non_k_matrix.shape[0]-temp_range[1]<k:
                temp_range[1]=non_k_matrix.shape[0]
        else:
            temp_range = [start, non_k_matrix.shape[0]]
        items_need_to_rate = set()
        for i in range(temp_range[0], temp_range[1]):
            items_need_to_rate = items_need_to_rate.union(supp_user(non_k_matrix[i, :]))
        items_need_to_rate = list(items_need_to_rate)
        for item_id in items_need_to_rate:
            for i in range(temp_range[0], temp_range[1]):
                if non_k_matrix[i][item_id] == 0:
                    non_k_matrix[i][item_id] = pd_rating(original_ratings, int(non_k_matrix[i][-1]), item_id,
                                                         [i for i in range(len(users))], 'trust', trust_web)
        start = temp_range[1]
        remain -= temp_range[1] - temp_range[0]

def test():
    global users,items,original_ratings
    items = [0, 1, 2, 3]
    users = [i for i in range(12)]
    original_ratings = np.array(
        [[0, 1, 4, 0], [0, 4, 1, 0], [4, 0, 0, 5], [4, 0, 3, 0], [4, 4, 0, 0], [4, 5, 0, 0], [0, 0, 4, 0], [0, 0, 5, 0],
         [0, 0, 3, 0], [0, 5, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]])
    ans=k_corate(2,original_ratings,np.ones(shape=(12,12)))
    logging.info(ans[0])
    logging.info(ans[1])

def k_corate(k,ratings,trust_web):
    sorted_ratings=sort(ratings)
    k_coreted,non_k_corated=part_k_corated(sorted_ratings,k)
    k_corating(k,non_k_corated,trust_web)
    ret=np.insert(k_coreted,k_coreted.shape[0],non_k_corated,0)
    ret_no_index=np.delete(ret,ret.shape[1]-1,1)
    return ret_no_index,ret


if __name__ == '__main__':
    k=10
    trust_web=np.loadtxt('trust_web.csv',delimiter=',')
    k_corated_ratings=k_corate(k,original_ratings,trust_web)[0]
    pd.DataFrame(k_corated_ratings).to_csv(str(k)+'_corated_ratings.csv', index=False, header=False)
