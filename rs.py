import numpy as np
import logging
import pandas as pd
import argparse
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
np.seterr(all='raise')
directory = 'ml-100k/'
rate_scale = 5
user_size = 943
item_size = 1682
trust_choices = ['default', 'adjust']
sim_choices = ['default', 'adjust']


def get_users(user_file):
    users = []
    logging.info('reading users from %s', user_file)
    with open(user_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            user_id, age, sex, occupation, zipcode = line.split('|')
            users.append(
                {'id': int(user_id), 'age': int(age), 'sex': sex, 'occupation': occupation, 'zipcode': zipcode})
    return users


def get_items(item_file):
    items = []
    logging.info('reading items from %s', item_file)
    with open(item_file, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
        for line in lines:
            movie_id, movie_title, release_date, video_release_date, IMDb_URL = line[:-39].split('|')
            genres = line[-38:]
            items.append({'id': int(movie_id), 'title': movie_title, 'release_date': release_date,
                          'video_release_date': video_release_date, 'IMDb_URL': IMDb_URL, 'genres': genres})
    return items


def get_ratings(rating_file):
    original_ratings = np.zeros(shape=(user_size, item_size))
    logging.info('reading ratings from %s', rating_file)
    with open(rating_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            user_id, item_id, rating, time_stamp = line.split('\t')
            original_ratings[int(user_id) - 1][int(item_id) - 1] = rating
    return original_ratings


def supp_user(user):
    return set(np.nonzero(user)[0])


def supp_item(item):
    return set(np.nonzero(item)[0])


def co_rated_items(user1, user2):
    return list(supp_user(user1).intersection(supp_user(user2)))


def co_bought_users(item1, item2):
    return list(supp_item(item1).intersection(supp_item(item2)))


def co_rated_mean(user1, user2):
    co_items = co_rated_items(user1, user2)
    sum1 = 0
    sum2 = 0
    for item in co_items:
        sum1 += user1[item]
        sum2 += user2[item]
    return sum1 / len(co_items), sum2 / len(co_items)


def single_mean(user):
    items = supp_user(user)
    sum_u = 0
    for item in items:
        sum_u += user[item]
    return sum_u / len(items)


def user_sim(user1, user2, mode):
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


def user_trust(ratings, id_user1, id_user2, mode):
    if mode == 'default':
        user1 = ratings[id_user1, :]
        user2 = ratings[id_user2, :]
        share_items = co_rated_items(user1, user2)
        if len(share_items) == 0:
            return 0
        temp_sum = 0
        for item_id in share_items:
            user1_mean, user2_mean = co_rated_mean(user1, user2)
            temp_sum += 1 - abs(user1_mean + user2[item_id] - user2_mean - user1[item_id]) / rate_scale
        return temp_sum / len(share_items)
    elif mode == 'adjust':
        pass
    else:
        raise ValueError


def user_corated(user1, user2):
    return len(co_rated_items(user1, user2))


def create_trust_web(original_ratings, mode, need_propogate):
    def trust_propagation(ratings, id_user1, id_user2, trust_web):
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

    trust_web = np.zeros(shape=(user_size, user_size))
    hundred = 1
    logging.info('start to fill trust web')
    for i in range(user_size):
        if i / 100 == hundred:
            logging.info('filling trust web, line %s00+', hundred)
            hundred += 1
        for j in range(user_size):
            if i == j:
                trust_web[i][j] = 1
            else:
                trust_web[i][j] = user_trust(original_ratings, i, j, mode)
    if need_propogate:
        hundred = 1
        logging.info('start to trust propagation')
        for i in range(user_size):
            if i / 100 == hundred:
                logging.info('trust propagating, line %s00+', hundred)
                hundred += 1
            for j in range(user_size):
                if trust_web[i][j] == 0:
                    trust_web[i][j] = trust_propagation(original_ratings, i, j, trust_web)
    return trust_web


def create_sim_web(original_ratings, mode):
    sim_web = np.zeros(shape=(user_size, user_size))
    hundred = 1
    logging.info('start to fill sim web')
    for i in range(user_size):
        if i / 100 == hundred:
            logging.info('filling sim web, line %s00+', hundred)
            hundred += 1
        for j in range(user_size):
            if i == j:
                sim_web[i, j] = 1
            else:
                sim_web[i, j] = user_sim(original_ratings[i, :], original_ratings[j, :], mode)
    return sim_web


def create_corated_web(original_ratings):
    corated_web = np.zeros(shape=(user_size, user_size), dtype=int)
    hundred = 1
    logging.info('start to fill corated web')
    for i in range(user_size):
        if i / 100 == hundred:
            logging.info('filling corated web, line %s00+', hundred)
            hundred += 1
        for j in range(user_size):
            corated_web[i, j] = user_corated(original_ratings[i, :], original_ratings[j, :])
    return corated_web


def pd_rating(ratings, user_id, item_id, web, nearest_neighbor_size):
    user = ratings[user_id, :]
    neighbor_ids = get_neighbors(ratings, user_id, item_id, web, nearest_neighbor_size)
    weight_sum = 0
    weight_dif_sum = 0
    single_user_mean = single_mean(user)
    for neighbor_id in neighbor_ids:
        neighbor = ratings[neighbor_id, :]
        if web[user_id][neighbor_id] != 0:
            neighbor_mean = co_rated_mean(user, neighbor)[1]
            weight_temp = web[user_id][neighbor_id]
            weight_sum += weight_temp
            weight_dif_sum += weight_temp * (ratings[neighbor_id, item_id] - neighbor_mean)
    try:
        assert weight_sum != 0
        return single_user_mean + weight_dif_sum / weight_sum
    except:
        return single_user_mean


def get_neighbors(ratings, user_id, item_id, web, nearest_neighbor_size):
    candi_neighbors = supp_item(ratings[:, item_id])
    candi_pairs = [(web[user_id, candi_neighbor], candi_neighbor) for candi_neighbor in candi_neighbors]
    candi_pairs.sort(key=lambda x: x[0], reverse=True)
    if nearest_neighbor_size > len(candi_pairs):
        return [x[1] for x in candi_pairs]
    else:
        return [candi_pairs[i][1] for i in range(nearest_neighbor_size)]


def dump(filename, matrix):
    pd.DataFrame(matrix).to_csv(filename + '.csv', index=False, header=False)
    logging.info('dump %s to csv', filename)


def main():
    # will create
    # [data_set]_ratings.csv
    # [data_set]_corated_web.csv
    # [data_set]_[trust_mode]_trust_prop_web.csv
    # [data_set]_[trust_mode]_trust_web.csv
    # [data_set]_[sim_mode]_sim_web.csv
    parser = argparse.ArgumentParser(description='Create ratings and webs of a certain file')
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-t', '--trust', choices=trust_choices, default='default')
    parser.add_argument('-s', '--sim', choices=sim_choices, default='default')
    args = parser.parse_args()
    data_set = args.dataset
    trust_mode = args.trust
    sim_mode = args.sim
    rating_file = directory + data_set + '.base'

    if not os.path.exists(data_set + '_ratings.csv'):
        original_ratings = get_ratings(rating_file)
        dump(data_set + '_ratings', original_ratings)
    else:
        original_ratings = np.loadtxt(data_set + '_ratings.csv', delimiter=',')
    if not os.path.exists(data_set + '_corated_web.csv'):
        corated_web = create_corated_web(original_ratings)
        dump(data_set + '_corated_web', corated_web)
    if trust_mode:
        trust_web = create_trust_web(original_ratings, trust_mode, True)
        dump(data_set + '_' + trust_mode + '_' + 'trust_prop_web', trust_web)
        trust_web = create_trust_web(original_ratings, trust_mode, False)
        dump(data_set + '_' + trust_mode + '_' + 'trust_web', trust_web)
    if sim_mode:
        sim_web = create_sim_web(original_ratings, sim_mode)
        dump(data_set + '_' + sim_mode + '_' + 'sim_web', sim_web)


if __name__ == '__main__':
    main()
