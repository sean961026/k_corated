from rs import rate_scale, user_size, supp_item, supp_user, item_size
import logging
import math
import random
import numpy as np
import argparse

method_choices=['best','dist']

def sim_rate(rate1, rate2):
    return math.exp(-abs(rate1 - rate2) / rate_scale)


def score(aux, record, ratings):
    item_ids = list(np.nonzero(aux)[0])
    sum = 0
    for item_id in item_ids:
        weight = 1 / supp_item(ratings[:, item_id])
        sum += weight * sim_rate(aux[item_id], record[item_id])
    return sum


def get_scores(aux, ratings):
    scores = []
    for i in range(user_size):
        record = ratings[i, :]
        scores.append(score(aux, record, ratings))
    std = np.std(scores)
    logging.info('the std of the scores is %s', std)
    return scores


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


def generate_aux(user, total, correct):
    items = list(supp_user(user))
    total_list = random.sample(items, total)
    correct_list = random.sample(total_list, correct)
    aux = np.zeros(shape=(1, item_size))
    for i in total_list:
        if i in correct_list:
            aux[i] = user[i]
        else:
            all_ratings = set([i + 1 for i in range(rate_scale)])
            avail_ratings = list(all_ratings - {user[i]})
            avail_ratings.sort()
            aux[i] = avail_ratings[random.randint(0, 3)]
    return aux


def show_how_similar(aux, record, ratings):
    logging.info(
        'the first line is the index of the co rated items, the second line is the ratings of the aux information, the third line is the ratings of the record')
    data = [i for i in range(item_size)]
    data.extend(aux)
    data.extend(record)
    arr = np.array(data)
    arr = np.reshape(arr, (3, item_size))
    to_delete = []
    for i in range(item_size):
        if arr[1, i] == 0 and arr[2, i] == 0:
            to_delete.append(i)
    arr = np.delete(arr, to_delete, 1)
    logging.info(arr)
    s = score(aux, record, ratings)
    logging.info('the score of the above 2 records is %s', s)


def ns_simulation(ratings_file_name, victim_id, total, correct, best_guess, param):
    logging.info('simulation of NS Attack to the victim %s in the rating file %s', victim_id, ratings_file_name)
    ratings = np.loadtxt(ratings_file_name, delimiter=',')
    victim = ratings[victim_id, :]
    aux = generate_aux(victim, total, correct)
    scores = get_scores(aux, ratings)
    if best_guess:
        best_id = de_anonymization(scores, param)
        if best_id:
            logging.info('found record %s most similar to the aux of %s with the eccentricity of ', best_id, victim_id,
                         param)
            show_how_similar(aux, ratings[best_id, :], ratings)
        else:
            logging.info('found no record qualified with the eccentricity of %s', param)
    else:
        dist = top_N_from_en(scores, param)
        for pro, index in dist:
            logging.info('found record %s similar to the aux of %s with the probability of %s', index, victim_id, pro)
            show_how_similar(aux, ratings[index, :], ratings)

def main():
    parser=argparse.ArgumentParser(description='a simulation of ns attack')
    parser.add_argument('-r','--ratings',required=True)
    parser.add_argument('-v','--victim',required=True,type=int)
    parser.add_argument('-t','--total',required=True,type=int)
    parser.add_argument('-c','--correct',required=True,type=int)
    parser.add_argument('-m','--method',required=True,choices=method_choices)
    parser.add_argument('-p','--param',required=True)
    args=parser.parse_args()
    ratings_file_name=args.ratings
    victim_id=args.victim
    total=args.total
    correct=args.total
    best_guess=args.method=='best'
    param=args.param
    ns_simulation(ratings_file_name,victim_id,total,correct,best_guess,param)

if __name__ == '__main__':
    main()