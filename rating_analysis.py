import matplotlib.pyplot as plt
from rs import load, get_ratings_name_from_dataset, supp_user
import argparse
import logging

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    args = parser.parse_args()
    original_ratings = load(get_ratings_name_from_dataset(args.dataset))
    analysis(original_ratings)
