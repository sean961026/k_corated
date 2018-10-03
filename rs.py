import numpy as np
import logging
import pandas as pd
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
np.seterr(all='raise')
directory = 'ml-100k/'
rate_scale = 5
trust_mode = 'default'
sim_mode = 'default'
weight_mode = 'default'
user_file = directory + 'u.user'
users = []
logging.info('reading users from %s', user_file)
with open(user_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        user_id, age, sex, occupation, zipcode = line.split('|')
        users.append(
            {'id': int(user_id), 'age': int(age), 'sex': sex, 'occupation': occupation, 'zipcode': zipcode})

item_file = directory + 'u.item'
items = []
logging.info('reading items from %s', item_file)
with open(item_file, 'r', encoding='ISO-8859-1') as file:
    lines = file.readlines()
    for line in lines:
        movie_id, movie_title, release_date, video_release_date, IMDb_URL = line[:-39].split('|')
        genres = line[-38:]
        items.append({'id': int(movie_id), 'title': movie_title, 'release_date': release_date,
                      'video_release_date': video_release_date, 'IMDb_URL': IMDb_URL, 'genres': genres})

rating_file = directory + sys.argv[-1]
original_ratings = np.zeros(shape=(len(users), len(items)))
logging.info('reading ratings from %s', rating_file)
with open(rating_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        user_id, item_id, rating, time_stamp = line.split('\t')
        original_ratings[int(user_id) - 1][int(item_id) - 1] = rating


def supp_user(user):
    return set(np.nonzero(user)[0])


def supp_item(item):
    return set(np.nonzero(item)[0])


def co_rated_items(user1, user2):
    return list(supp_user(user1).intersection(supp_user(user2)))


def co_bought_users(item1, item2):
    return list(supp_item(item1).intersection(supp_item(item2)))


def user_sim(user1, user2, mode='default'):
    if mode == 'default':
        share_items = co_rated_items(user1, user2)
        temp_user_1 = [user1[i] for i in share_items]
        temp_user_2 = [user2[i] for i in share_items]
        if len(temp_user_1) == 0:
            return 0
        if len(set(temp_user_2)) == 1 or len(set(temp_user_1)) == 1:
            return 0
        return np.corrcoef(temp_user_1, temp_user_2)[0, 1]
    elif mode == 'adjust':
        pass
    else:
        raise ValueError


def pd_rating(ratings, user_id, item_id, neighbor_ids, mode='default',trust_web=None):
    user = ratings[user_id, :]
    if len(neighbor_ids) == 1:
        neighbor = ratings[neighbor_ids[0], :]
        return user.mean() + neighbor[item_id] - neighbor.mean()
    weight_sum = 0
    weight_dif_sum = 0
    for neighbor_id in neighbor_ids:
        neighbor = ratings[neighbor_id,:]
        if mode == 'default':
            weight_temp = user_sim(user,neighbor)
        elif mode == 'trust':
            weight_temp = trust_web[user_id][neighbor_id]
        else:
            raise ValueError
        weight_sum += weight_temp
        weight_dif_sum += weight_temp * (ratings[neighbor_id, item_id] - neighbor.mean())
    try:
        assert weight_sum != 0
        return user.mean() + weight_dif_sum / weight_sum
    except:
        logging.warning('unexpected weight_sum==0')
        return user.mean()


def user_trust(ratings, id_user1, id_user2, mode='default'):
    if mode == 'default':
        user1 = ratings[id_user1, :]
        user2 = ratings[id_user2, :]
        share_items = co_rated_items(user1, user2)
        if len(share_items) == 0:
            return 0
        temp_sum = 0
        for item_id in share_items:
            temp_sum += 1 - abs(
                pd_rating(ratings, id_user1, item_id, [id_user2]) - ratings[id_user1, item_id]) / rate_scale
        return temp_sum / len(share_items)
    elif mode == 'adjust':
        pass
    else:
        raise ValueError


def trust_propagation(ratings, id_user1, id_user2):
    temp_mid = set()
    items_bought_by_user1 = supp_user(ratings[id_user1, :])
    for item_id in items_bought_by_user1:
        users_buy_such_item = supp_item(ratings[:, item_id])
        if id_user2 in users_buy_such_item:
            for temp_user in users_buy_such_item:
                if temp_user != id_user1 and temp_user != id_user2:
                    temp_mid.add(temp_user)
    up = 0
    down = 0
    for mid in temp_mid:
        trust_1_m = trust_web[id_user1][mid]
        trust_m_2 = trust_web[mid][id_user2]
        items_1_m = len(co_rated_items(id_user1, mid))
        items_m_2 = len(co_rated_items(mid, id_user2))
        up += items_1_m * trust_1_m + items_m_2 * trust_m_2
        down += items_1_m + items_m_2
    if len(temp_mid) == 0:
        return 0
    return up / down


if __name__ == '__main__':
    trust_web = np.zeros(shape=(len(users), len(users)))
    hundred = 1
    logging.info('start to fill trust web')
    for i in range(len(users)):
        if i / 100 == hundred:
            logging.info('filling trust web, line %s00+', hundred)
            hundred += 1
        for j in range(len(users)):
            if i == j:
                trust_web[i][j] = 1
            else:
                trust_web[i][j] = user_trust(original_ratings, i, j, trust_mode)
    hundred = 1
    logging.info('start to trust propagation')
    for i in range(len(users)):
        if i / 100 == hundred:
            logging.info('trust propagating, line %s00+', hundred)
            hundred += 1
        for j in range(len(users)):
            if trust_web[i][j] == 0:
                trust_web[i][j] = trust_propagation(original_ratings, i, j)
    pd.DataFrame(original_ratings).to_csv(sys.argv[-1]+'_ratings.csv', index=False, header=False)
    pd.DataFrame(trust_web).to_csv(sys.argv[-1]+'_trust_web.csv', index=False, header=False)
