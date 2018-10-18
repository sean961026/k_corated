from rs import rating_scale, supp_item, supp_user, load, extract_dataset_from_filename, get_ratings_name_from_dataset, \
    unknown_rating
from k_corated_by import get_index_from_krating_file
import logging
import math
import random
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

mode_choices = ['exp', 'indicate']
sim_threshold = 0
sim_mode = ''
weight_mode = ''
attack_ratings = None
original_ratings = None
user_size = 0
item_size = 0
translator = []
o2k = lambda x: translator.index(x)
pro_dist_count = 0


def get_id_translator(attack_ratings_file_name):
    index_file = get_index_from_krating_file(attack_ratings_file_name)
    if not os.path.exists(index_file):
        return [i for i in range(user_size)]
    index = np.loadtxt(index_file)
    index_data = [int(i) - 1 for i in index]
    return index_data


def sim_rate(rate1, rate2):
    if rate1 == unknown_rating or rate2 == unknown_rating:
        return 0
    if sim_mode == 'exp':
        return math.exp(-abs(rate1 - rate2) / rating_scale)
    elif sim_mode == 'indicate':
        return 1 if abs(rate1 - rate2) <= sim_threshold else 0
    else:
        raise ValueError


def item_weight(item_id):
    if weight_mode == 'equal':
        return 1
    elif weight_mode == 'less':
        return 1 / len(supp_item(attack_ratings[:, item_id]))
    else:
        raise ValueError


def score(aux, record):
    item_ids = supp_user(aux)
    sum = 0
    for item_id in item_ids:
        weight = item_weight(item_id)
        sum += weight * sim_rate(aux[item_id], record[item_id])
    return sum


def get_scores(aux):
    scores = []
    for i in range(user_size):
        record = attack_ratings[i, :]
        scores.append(score(aux, record))
    return scores


def analyze_scores(scores):
    std = np.std(scores)
    temp = scores.copy()
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    threshold = (max1 - max2) / std
    return {'max1': max1, 'max2': max2, 'std': std, 'threshold': threshold}


def generate_aux(user_id, total, correct):
    def random_wrong(start, end, right_value, tp):
        if tp == int:
            while True:
                r = random.randint(start, end)
                if r != right_value:
                    return r
        elif tp == float:
            while True:
                r = random.uniform(start, end)
                if r != right_value:
                    return r

    user = original_ratings[user_id, :]
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
            aux[i] = random_wrong(1, 5, user[i], int)
    return aux


def generate_auxs(total, correct):
    logging.info('generating auxs from ratings')
    auxs = [0] * original_ratings.shape[0]
    for i in range(original_ratings.shape[0]):
        aux = generate_aux(i, total, correct)
        auxs[o2k(i)] = aux
    return auxs


def de_anonymization(scores, eccen):
    data = analyze_scores(scores)
    threshold = data['threshold']
    if threshold < eccen:
        return None
    else:
        best_id = scores.index(data['max1'])
        return best_id


def entropic_de(scores):
    data = analyze_scores(scores)
    std = data['std']
    temp = [math.exp(i / std) for i in scores]
    total = sum(temp)
    c = 1 / total
    dist = [(c * temp[i], i) for i in range(len(temp))]
    return dist


def top_N_from_en(scores, N):
    dist = entropic_de(scores)
    dist.sort(key=lambda x: x[0], reverse=True)
    logging.info(sa2scores(scores))
    logging.info(sa2dist(dist))
    return [dist[i] for i in range(N)]


def de_attack_2_range(auxs, eccen, rg):
    logging.info('attacking the ratings by best guess')
    result = []
    for i in rg:
        aux = auxs[i]
        scores = get_scores(aux)
        ans = de_anonymization(scores, eccen)
        threshold = analyze_scores(scores)['threshold']
        if ans is None:
            result.append(('no_match', threshold))
        else:
            if ans == i:
                result.append(('match_success', threshold))
            else:
                result.append(('match_failure', threshold))
    return result


def en_attack_2_range(auxs, N, rg):
    logging.info('attacking the ratings by distribution')
    dists = []
    for i in rg:
        aux = auxs[i]
        scores = get_scores(aux)
        dist = top_N_from_en(scores, N)
        dists.append(dist)
    return dists


def sa2scores(scores):
    scores.sort(reverse=True)
    x = [i for i in range(len(scores))]
    y = [score for score in scores]
    plt.figure()
    plt.plot(x, y)
    plt.savefig('score_dist_%s.jpg' % pro_dist_count)
    std = np.std(scores)

    def group_diff(diff):
        for i in range(len(scores) - 1):
            if (scores[i] - scores[i + 1]) / std >= diff:
                return i
        return None

    temp = [(scores[i] - scores[i + 1]) / std for i in range(len(scores) - 1)]

    return {'eccen1': group_diff(1), 'eccen1.5': group_diff(1.5), 'max_diff': max(temp)}




def sa2dist(dist):
    global pro_dist_count

    def percent2size(percent):
        up_limit = len(dist)
        while up_limit > 0:
            temp = [dist[i][0] for i in range(up_limit)]
            s2 = sum(temp)
            s1 = s2 - temp.pop()
            if s1 < percent and s2 >= percent:
                return len(temp)
            else:
                up_limit -= 1
        return 0

    def size2propsum(size):
        temp = [dist[i][0] for i in range(size)]
        return sum(temp)

    def top_group(size):
        return [dist[i][1] for i in range(size)]

    x = [i for i in range(len(dist))]
    y = [dist[i][0] for i in range(len(dist))]
    plt.figure()
    plt.plot(x, y)
    plt.savefig('pro_dist_%s.jpg' % pro_dist_count)
    pro_dist_count += 1
    analysis_data = {'max_pro': dist[0][0], '90p': percent2size(0.9), '95p': percent2size(0.95),
                     'top10': size2propsum(10), 'top20': size2propsum(20), 'group5': top_group(5),
                     'group10': top_group(10)}
    return analysis_data


def sa2de_all(result):
    logging.info('analyzing the result of best guess')
    success_list = []
    no_match_list = []
    wrong_match_list = []
    thresholds = []
    for i in range(len(result)):
        threshold = result[i][1]
        thresholds.append(threshold)
        if result[i][0] == 'no_match':
            no_match_list.append(i)
        else:
            if result[i][0] == 'match_failure':
                wrong_match_list.append(i)
            else:
                success_list.append(i)

    min_int_threshold = int(min(thresholds))
    max_int_threshold = int(max(thresholds)) + 1
    bins = [i / 2 for i in range(min_int_threshold * 2, max_int_threshold * 2 + 1)]
    plt.hist(thresholds, bins=bins)
    plt.savefig('threshold.jpg')
    return {'success_rate': len(success_list) / len(result), 'no_match_rate': len(no_match_list) / len(result),
            'wrong_match_rate': len(wrong_match_list) / len(result),
            'attack_size': len(result)}, no_match_list, success_list, wrong_match_list


def statistical_analysis(auxs, eccen, N):
    if eccen and N:
        result = de_attack_2_range(auxs, eccen, rg=range(user_size))
        analysis_data, no_match_list, success_match_list, wrong_match_list = sa2de_all(result)
        logging.info(analysis_data)
        logging.info('analyzing the result of distribution on those best-guess-failure cases')
        wrong_list = no_match_list + wrong_match_list
        dists = en_attack_2_range(auxs, N, rg=[wrong_list[i] for i in range(5)])
    elif eccen and N is None:
        result = de_attack_2_range(auxs, eccen, rg=range(user_size))
        analysis_data, no_match_list, success_match_list, wrong_match_list = sa2de_all(result)
        logging.info(analysis_data)


def init(_sim_threshold, _sim_mode, attack_ratings_file, _weight_mode):
    global sim_threshold, sim_mode, attack_ratings, original_ratings, translator, user_size, item_size, weight_mode
    weight_mode = _weight_mode
    sim_threshold = _sim_threshold
    sim_mode = _sim_mode
    attack_ratings = load(attack_ratings_file)
    dataset = extract_dataset_from_filename(attack_ratings_file)
    original_ratings = load(get_ratings_name_from_dataset(dataset))
    user_size = original_ratings.shape[0]
    item_size = original_ratings.shape[1]
    translator = get_id_translator(attack_ratings_file)


def main():
    parser = argparse.ArgumentParser(description='a ns attack simulation and statitical analysis')
    parser.add_argument('-r', '--ratings', required=True)
    parser.add_argument('-t', '--total', required=True, type=int)
    parser.add_argument('-c', '--correct', required=True, type=int)
    parser.add_argument('-e', '--eccen', type=float)
    parser.add_argument('-n', type=int)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('-m,', '--mode', choices=mode_choices)
    parser.add_argument('-w', '--weight', choices=['equal', 'less'], default='less')
    args = parser.parse_args()
    init(args.threshold, args.mode, args.ratings, args.weight)
    auxs = generate_auxs(args.total, args.correct)
    statistical_analysis(auxs, args.eccen, args.n)


if __name__ == '__main__':
    main()
