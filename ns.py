from rs import rate_scale, user_size, supp_item, supp_user, item_size
import logging
import math
import random
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from dist import hellinger_distance, get_prop_dist_global, get_prop_dist_local

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


def generate_auxs(ratings, total, correct):
    logging.info('generating auxs from ratings')
    auxs = []
    for i in range(ratings.shape[0]):
        aux = generate_aux(ratings, i, total, correct)
        auxs.append(aux)
    return auxs


def analyze_scores(scores):
    std = np.std(scores)
    temp = scores.copy()
    max1 = max(temp)
    temp.remove(max1)
    max2 = max(temp)
    threshold = (max1 - max2) / std
    return {'max1': max1, 'max2': max2, 'std': std, 'threshold': threshold}


def de_anonymization(scores, eccen):
    data = analyze_scores(scores)
    threshold = data['threshold']
    if threshold < eccen:
        return None
    else:
        best_id = scores.index(data['max1'])
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


def de_attack_2_all(ratings, auxs, eccen):
    logging.info('attacking the ratings by best guess')
    record_size = ratings.shape[0]
    result = []
    for i in range(record_size):
        aux = auxs[i]
        scores = get_scores(aux, ratings)
        ans = de_anonymization(scores, eccen)
        threshold = analyze_scores(scores)['threshold']
        if ans is None:
            result.append((None, threshold))
        else:
            if ans == i:
                result.append((1, threshold))
            else:
                result.append((0, threshold))
    return result


def en_attack_2_all(ratings, auxs, N):
    logging.info('attacking the ratings by distribution')
    dists = []
    record_size = ratings.shape[0]
    for i in range(record_size):
        aux = auxs[i]
        scores = get_scores(aux, ratings)
        dist = top_N_from_en(scores, N)
        dists.append(dist)
    return dists


def sa2dist(dist, ratings, target_record, **analysis_data):
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
    base_score = score(target_record, target_record, ratings)
    fitting_score = score(result, target_record, ratings)
    distance = prop_distantce_of_dist(dist, ratings, target_record)
    analysis_data.update(
        {'base_score': base_score, 'fitting_score': fitting_score, 'percent': fitting_score / base_score,
         'distance': distance})
    return analysis_data


def prop_distantce_of_dist(dist, ratings, target_record):
    items = supp_user(target_record)
    local = [i[1] for i in dist]
    dist_all = 0
    for item_id in items:
        glo = get_prop_dist_global(ratings[:, item_id])
        loc = get_prop_dist_local(ratings[:, item_id], local)
        dist_all += hellinger_distance(glo, loc)
    return dist_all / len(items)


def get_id_translator(attack_ratings_file_name):
    index_file = attack_ratings_file_name[:-4] + '_index.csv'
    if not os.path.exists(index_file):
        return [i for i in range(user_size)]
    index = np.loadtxt(index_file)
    index_data = [int(i) - 1 for i in index]
    return index_data


def sa2de_all(result):
    logging.info('analyzing the result of best guess')
    success_list = []
    no_match_list = []
    wrong_match_list = []
    thresholds = []
    for i in range(len(result)):
        threshold = result[i][1]
        thresholds.append(threshold)
        if result[i][0] == None:
            no_match_list.append(i)
        else:
            if result[i][0] == 0:
                wrong_match_list.append(i)
            else:
                success_list.append(i)

    min_int_threshold = int(min(thresholds))
    max_int_threshold = int(max(thresholds)) + 1
    bins = [i / 2 for i in range(min_int_threshold * 2, max_int_threshold * 2 + 1)]
    plt.hist(thresholds, bins=bins)
    plt.savefig('threshold.jpg')
    max_no_match_ts = result[max(no_match_list, key=lambda x: result[x][1])][1] if no_match_list else None
    min_sucess_ts = result[min(success_list, key=lambda x: result[x][1])][1] if success_list else None
    return {'min_sucess_ts': min_sucess_ts, 'max_no_match_ts': max_no_match_ts,
            'success_rate': len(success_list) / len(result), 'no_match_rate': len(no_match_list) / len(result),
            'wrong_match_rate': len(wrong_match_list) / len(result), 'attack_size': len(result)}


def statistical_analysis(ratings, auxs, eccen, N):
    if eccen and N:
        result = de_attack_2_all(ratings, auxs, eccen)
        analysis_data = sa2de_all(result)
        logging.info(analysis_data)
        dists = en_attack_2_all(ratings, auxs, N)
        logging.info('analyzing the result of distribution on those best-guess-failure cases')
        for i in range(len(result)):
            success = result[i][0]
            threshold = result[i][1]
            if success != 1:
                if success == 0:
                    reason = 'wrong_match'
                else:
                    reason = 'no_match'
                target_record = ratings[i, :]
                dist = dists[i]
                analysis_data = sa2dist(dist, ratings, target_record,
                                        **{'failure_reason': reason, 'failure_ts': threshold})
                logging.info(analysis_data)
                sys.stdout.flush()
    elif eccen and N is None:
        result = de_attack_2_all(ratings, auxs, eccen)
        analysis_data = sa2de_all(result)
        logging.info(analysis_data)


def main():
    parser = argparse.ArgumentParser(description='a ns attack simulation and statitical analysis')
    parser.add_argument('-r', '--ratings', required=True)
    parser.add_argument('-t', '--total', required=True, type=int)
    parser.add_argument('-c', '--correct', required=True, type=int)
    parser.add_argument('-e', '--eccen', type=float)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    ratings_to_attack = np.loadtxt(args.ratings, delimiter=',')
    dataset_list = ['u1', 'u2', 'u3', 'u4', 'u5']
    for dataset in dataset_list:
        if dataset in args.ratings:
            break
    else:
        logging.info('illegal ratings file')
        return
    ratings_to_gen_aux = np.loadtxt(dataset + '_ratings.csv', delimiter=',')
    total = args.total
    correct = args.correct
    eccen = args.eccen
    N = args.n
    temp_auxs = generate_auxs(ratings_to_gen_aux, total, correct)
    translator = get_id_translator(args.ratings)
    auxs = [0] * len(temp_auxs)
    for i in range(len(temp_auxs)):
        index_in_attack = translator.index(i)
        auxs[index_in_attack] = temp_auxs[i]
    statistical_analysis(ratings_to_attack, auxs, eccen, N)


if __name__ == '__main__':
    main()
