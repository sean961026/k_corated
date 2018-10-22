from rs import rating_scale, supp_item, supp_user, load, extract_dataset_from_filename, get_ratings_name_from_dataset, \
    unknown_rating
from k_corated_by import get_index_from_krating_file
from dist import hellinger_distance, get_prop_dist_from_ratings
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
k2o = lambda x: translator[x]
eccen = 0
total = 0
correct = 0
distance_pic_count = 0


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


def generate_aux(user_id):
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


def generate_auxs():
    logging.info('generating auxs from ratings')
    auxs = [0] * original_ratings.shape[0]
    for i in range(original_ratings.shape[0]):
        aux = generate_aux(i)
        auxs[o2k(i)] = aux
    return auxs


def de_anonymization(scores):
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


def de_attack_2_range(auxs, rg):
    logging.info('attacking the ratings by best guess')
    result = []
    for i in rg:
        aux = auxs[i]
        scores = get_scores(aux)
        ans = de_anonymization(scores)
        threshold = analyze_scores(scores)['threshold']
        if ans is None:
            result.append(('no_match', threshold))
        else:
            if ans == i:
                result.append(('match_success', threshold))
            else:
                result.append(('match_failure', threshold))
    return result


def en_attack_2_range(auxs, rg):
    logging.info('attacking the ratings by distribution')
    for i in rg:
        aux = auxs[i]
        scores = get_scores(aux)
        dist = entropic_de(scores)
        sa2en_attack(i, scores, dist)


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


def candi_list(scores):
    candidates = []
    for i in range(len(scores)):
        if scores[i] >= correct:
            candidates.append(i)
    return candidates


def sa2en_attack(attackee, scores, dist):
    global distance_pic_count
    dist.sort(key=lambda x: x[0], reverse=True)

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

    candi = candi_list(scores)
    attackee_record = attack_ratings[attackee, :]
    items_id = supp_user(attackee_record)
    x = [i for i in range(items_id)]
    y = []
    for item_id in items_id:
        global_ratings = supp_item(attack_ratings[:, item_id])
        gp = get_prop_dist_from_ratings(global_ratings)
        local_ratings = [global_ratings[i] for i in candi]
        lp = get_prop_dist_from_ratings(local_ratings)
        y.append(hellinger_distance(gp, lp))
    plt.figure()
    plt.plot(x, y)
    plt.savefig('dist_%s.jpg' % distance_pic_count)
    distance_pic_count += 1
    analysis_data = {'attackee': attackee, 'group10': top_group(10), 'top10': size2propsum(10), 'max_pro': dist[0][0],
                     'candidates_size': len(candi), 'candi_prop': size2propsum(len(candi))}
    return analysis_data


def statistical_analysis(auxs):
    if eccen:
        result = de_attack_2_range(auxs, rg=range(user_size))
        analysis_data, no_match_list, success_match_list, wrong_match_list = sa2de_all(result)
        logging.info(analysis_data)
        logging.info('analyzing the result of distribution on those best-guess-failure cases')
        wrong_list = no_match_list + wrong_match_list
        en_attack_2_range(auxs, rg=[wrong_list[i] for i in range(5)])


def init():
    global weight_mode, sim_threshold, sim_mode, attack_ratings, original_ratings, user_size, item_size, translator, correct, total, eccen
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
    weight_mode = args.weight
    sim_threshold = args.threshold
    sim_mode = args.mode
    attack_ratings = load(args.ratings)
    dataset = extract_dataset_from_filename(args.ratings)
    original_ratings = load(get_ratings_name_from_dataset(dataset))
    user_size = original_ratings.shape[0]
    item_size = original_ratings.shape[1]
    translator = get_id_translator(args.ratings)
    correct = args.correct
    total = args.total
    eccen = args.eccen


def main():
    init()
    auxs = generate_auxs()
    statistical_analysis(auxs)


if __name__ == '__main__':
    main()
