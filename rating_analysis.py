import matplotlib.pyplot as plt
from rs import load, get_ratings_name_from_dataset, supp_user, rating_scale
import argparse
import logging
import numpy as np

def analysis(original_ratings):
    size_list = []
    for user in original_ratings:
        size = len(supp_user(user))
        size_list.append(size)
    size_list.sort()
    x = [i for i in range(len(size_list))]
    plt.figure()
    plt.plot(x, size_list)
    plt.savefig('size.jpg')
    portion = [0] * 10
    for size in size_list:
        index = size // 100
        portion[index] += 1
    logging.info(portion)
    logging.info(sum(size_list) / len(size_list))


def ratia_analysis(original_ratings):
    def get_count(vector):
        count = [0] * rating_scale
        for i in vector:
            index = int(i) - 1
            count[index] += 1
        return count

    all_count = [0] * rating_scale
    items_count = []
    item_size = original_ratings.shape[1]
    for i in range(item_size):
        item = original_ratings[:, i]
        temp_count = get_count(item)
        items_count.append(temp_count)
        all_count = list(np.array(all_count) + np.array(temp_count))
    # for i in range(rating_scale):
    #     plt.figure()
    #     label = 'portion of %s' % (i + 1)
    #     x = [i for i in range(item_size)]
    #     y = [temp_count[i] / sum(temp_count) for temp_count in items_count]
    #     plt.plot(x, y, label=label)
    #     plt.legend()
    #     plt.savefig('items_portion_%s.jpg' % (i + 1))
    logging.info([i / sum(all_count) for i in all_count])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ratings', required=True)
    args = parser.parse_args()
    original_ratings = load(args.ratings)
    ratia_analysis(original_ratings)
