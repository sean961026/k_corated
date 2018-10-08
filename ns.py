from rs import rate_scale, user_size, supp_item, supp_user, item_size
import logging
import math
import random
import numpy as np
import argparse
import os

method_choices = ['best', 'dist']


def sim_rate(rate1, rate2):
    return math.exp(-abs(rate1 - rate2) / rate_scale)


def score(aux, record, ratings):
    item_ids = list(np.nonzero(aux)[0])
    sum = 0
    for item_id in item_ids:
        weight = 1 / len(supp_item(ratings[:, item_id]))
        sum += weight * sim_rate(aux[item_id], record[item_id])
    return sum


def get_scores(aux, ratings):
    scores = []
    for i in range(user_size):
        record = ratings[i, :]
        scores.append(score(aux, record, ratings))
    std = np.std(scores)
    logging.debug('the std of the scores is %s', std)
    temp = scores.copy()
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    threshold = (max1 - max2) / std
    logging.info('(max1-max2)/std is %s', threshold)
    return scores


def de_anonymization(scores, eccen):
    temp = scores.copy()
    std = np.std(temp)
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    logging.info('(max1-max2)/std is %s', (max1 - max2) / std)
    if (max1 - max2) / std < eccen:
        return None
    else:
        best_id = scores.index(max1)
        return best_id


def entropic_de(scores):
    std = np.std(scores)
    temp = [math.exp(i / std) for i in scores]
    total = sum(temp)
    c = 1 / total
    dist = [(c * temp[i], i) for i in range(len(temp))]
    return dist


def top_N_from_en(scores, N):
    dist = entropic_de(scores)
    dist.sort(key=lambda x: x[0], reverse=True)
    return [dist[i] for i in range(N)]


def generate_aux(user, total, correct):
    items = list(supp_user(user))
    if total < len(items):
        total_list = random.sample(items, total)
    else:
        total_list = items
    if correct < len(total_list):
        correct_list = random.sample(total_list, correct)
    else:
        correct_list = total_list
    aux = [0] * item_size
    for i in total_list:
        if i in correct_list:
            aux[i] = user[i]
        else:
            all_ratings = set([i + 1 for i in range(rate_scale)])
            avail_ratings = list(all_ratings - {user[i]})
            avail_ratings.sort()
            aux[i] = avail_ratings[random.randint(0, 3)]
    return np.array(aux)


def ns_simulation(ratings_file_name, victim_id, aux, best_guess, param):
    logging.info('simulation of NS Attack to the victim %s in the rating file %s', victim_id, ratings_file_name)
    ratings = np.loadtxt(ratings_file_name, delimiter=',')
    scores = get_scores(aux, ratings)
    if best_guess:
        best_id = de_anonymization(scores, param)
        if best_id:
            logging.info('found record %s most similar to the aux of %s with the eccentricity of ', best_id, victim_id,
                         param)
        else:
            logging.info('found no record qualified with the eccentricity of %s', param)
    else:
        dist = top_N_from_en(scores, param)
        for pro, index in dist:
            logging.info('record:%s\tpropability:%s', index, pro)


def simulation():
    parser = argparse.ArgumentParser(description='a simulation of ns attack')
    parser.add_argument('-r', '--ratings', required=True)
    parser.add_argument('-v', '--victim', required=True, type=int)
    parser.add_argument('-t', '--total', required=True, type=int)
    parser.add_argument('-c', '--correct', required=True, type=int)
    parser.add_argument('-m', '--method', required=True, choices=method_choices)
    parser.add_argument('-p', '--param', required=True)
    parser.add_argument('-k', '--kratings')
    args = parser.parse_args()
    ratings_file_name = args.ratings
    victim_id = args.victim
    total = args.total
    correct = args.total
    best_guess = args.method == 'best'
    k_corated_ratings_file = args.kratings
    param = args.param
    if best_guess:
        param = float(param)
    else:
        param = int(param)
    original_ratings = np.loadtxt(ratings_file_name, delimiter=',')
    aux = generate_aux(original_ratings[victim_id, :], total, correct)
    if k_corated_ratings_file:
        compare(ratings_file_name, k_corated_ratings_file, victim_id, aux, best_guess, param)
    else:
        ns_simulation(ratings_file_name, victim_id, aux, best_guess, param)


def compare(original_ratings_file, k_corated_ratings_file, victim_id, aux, best_guess, param):
    ns_simulation(original_ratings_file, victim_id, aux, best_guess, param)
    victim_id_in_k = id_transfer(k_corated_ratings_file, victim_id)
    ns_simulation(k_corated_ratings_file, victim_id_in_k, aux, best_guess, param)


def id_transfer(k_corated_ratings_file, victim_id):
    index_file = k_corated_ratings_file[:-4] + '_index.csv'
    if not os.path.exists(index_file):
        return victim_id
    index = np.loadtxt(index_file)
    index_data = [int(i) - 1 for i in index]
    victim_id_in_k = index_data.index(victim_id)
    return victim_id_in_k


def sa2best_guess(original_ratings_file, k_corated_ratings_file, total, correct, eccen, sample_size):
    success = 0
    original_ratings = np.loadtxt(original_ratings_file, delimiter=',')
    k_corated_ratings = np.loadtxt(k_corated_ratings_file, delimiter=',')
    sample = random.sample(range(user_size), sample_size)
    for i in sample:
        aux = generate_aux(original_ratings[i, :], total, correct)
        scores = get_scores(aux, k_corated_ratings)
        ans = de_anonymization(scores, eccen)
        id_in_k = id_transfer(k_corated_ratings_file, i)
        if ans == id_in_k:
            success += 1
    ret = success / user_size
    logging.info('the probability of one sample is %s', ret)
    return ret


def statistical_analysis():
    parser = argparse.ArgumentParser(description='a statistical analysis of ns attack')
    parser.add_argument('-r', '--ratings', required=True)
    parser.add_argument('-t', '--total', required=True, type=int)
    parser.add_argument('-c', '--correct', required=True, type=int)
    parser.add_argument('-m', '--method', required=True, choices=method_choices)
    parser.add_argument('-p', '--param', required=True)
    parser.add_argument('-k', '--kratings', required=True)
    parser.add_argument('-s', '--sample', type=int, default=200)
    parser.add_argument('--trial', type=int, default=3)
    args = parser.parse_args()
    ratings_file_name = args.ratings
    total = args.total
    correct = args.total
    best_guess = args.method == 'best'
    k_corated_ratings_file = args.kratings
    param = args.param
    sample_size = args.sample
    trial_size = args.trial
    if best_guess:
        param = float(param)
    else:
        param = int(param)
    result = 0
    for i in range(trial_size):
        result += sa2best_guess(ratings_file_name, k_corated_ratings_file, total, correct, param, sample_size)
    logging.info('the probability is %s', result / trial_size)


if __name__ == '__main__':
    statistical_analysis()
