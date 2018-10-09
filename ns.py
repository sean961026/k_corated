from rs import rate_scale, user_size, supp_item, supp_user, item_size
import logging
import math
import random
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

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
    return scores


def analyze_scores(scores):
    std = np.std(scores)
    temp = scores.copy()
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    threshold = (max1 - max2) / std
    return {'max1': max1, 'max2': max2, 'std': std, 'threshold': threshold}


def de_anonymization(scores, eccen):
    temp = scores.copy()
    std = np.std(temp)
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
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


def generate_aux(original_ratings, user_id, total, correct):
    user = original_ratings[user_id, :]
    items = list(supp_user(user))
    if total < len(items):
        total_list = random.sample(items, total)
    else:
        logging.debug('cannot sample,size is %s', len(items))
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


def de_attack_2_all(ratings, total, correct, eccen):
    record_size = ratings.shape[0]
    result = [(0, 0)] * record_size
    for i in range(record_size):
        aux = generate_aux(ratings, i, total, correct)
        scores = get_scores(aux, ratings)
        ans = de_anonymization(scores, eccen)
        if ans == i:
            result[i] = (1, analyze_scores(scores)['threshold'])
    return result


def en_attack_2_all(ratings, total, correct, N):
    dists = []
    record_size = ratings.shape[0]
    for i in range(record_size):
        aux = generate_aux(ratings, i, total, correct)
        scores = get_scores(aux, ratings)
        dist = top_N_from_en(scores, N)
        dists.append(dist)
    return dists


def sa2dist(dist, ratings, target_record):
    items = supp_user(target_record)
    result = [0] * item_size
    for item_id in items:
        up_sum = 0
        down_sum = 0
        for prop, index in dist:
            record = ratings[index, :]
            if record[item_id] != 0:
                up_sum += prop * record[item_id]
                down_sum += prop
        if down_sum != 0:
            result[item_id] = up_sum / down_sum
    return result


def id_transfer(k_corated_ratings_file, victim_id):
    index_file = k_corated_ratings_file[:-4] + '_index.csv'
    if not os.path.exists(index_file):
        return victim_id
    index = np.loadtxt(index_file)
    index_data = [int(i) - 1 for i in index]
    victim_id_in_k = index_data.index(victim_id)
    return victim_id_in_k


def sa2de_all(result):
    success_list = [i[0] for i in result]
    thresholds = [i[1] for i in result]
    success = sum(success_list)
    sucess_rate = success / len(result)
    success_thresholds = []
    failure_thresholds = []
    for suc, threshold in result:
        if suc:
            success_thresholds.append(threshold)
        else:
            failure_thresholds.append(threshold)
    min_sucess_threshold = min(success_thresholds)
    max_failure_threshold = max(failure_thresholds)
    min_int_threshold = int(min(thresholds))
    max_int_threshold = int(max(thresholds)) + 1
    bin = [i / 2 for i in range(min_int_threshold * 2, max_int_threshold * 2 + 1)]
    plt.hist(thresholds, bin=bin)
    plt.savefig('threshold.jpg')
    return {'min_sucess_threshold': min_sucess_threshold, 'max_failure_threshold': max_failure_threshold,
            'success_rate': sucess_rate}


def statistical_analysis(ratings, total, correct, eccen, N):
    result = de_attack_2_all(ratings, total, correct, eccen)
    dists = en_attack_2_all(ratings, total, correct, N)
    analysis_data = sa2de_all(result)
    logging.info(analysis_data)
    for i in range(len(result)):
        sucess = result[i][0]
        threshold = result[i][1]
        if not sucess:
            target_record = ratings[i, :]
            dist = dists[i]
            result = sa2dist(dist, ratings, target_record)
            base_score = score(target_record, target_record, ratings)
            result_score = score(result, target_record, ratings)
            logging.info('failed_index:%s\tfailed_threshold:%s\tself_score:%s\tresult_score:%s\tpercent:%s', i,
                         threshold, base_score, result_score, result_score / base_score)


def main():
    parser = argparse.ArgumentParser(description='a ns attack simulation and statitical analysis')
    parser.add_argument('-r', '--ratings', required=True)
    parser.add_argument('-t', '--total', required=True, type=int)
    parser.add_argument('-c', '--correct', required=True, type=int)
    parser.add_argument('-e', '--eccen', type=float, required=True)
    parser.add_argument('-n', type=int, required=True)
    args = parser.parse_args()
    ratings = np.loadtxt(args.ratings, delimiter=',')
    total = args.total
    correct = args.correct
    eccen = args.eccen
    N = args.n
    statistical_analysis(ratings, total, correct, eccen, N)

if __name__ == '__main__':
    main()
