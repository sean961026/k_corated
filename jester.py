import numpy as np
from rs import unknown_rating, load, dump
import argparse
import random

max_user_size = 63978


def create_ratings(user_size, filename, mode):
    ratings = np.zeros(shape=(user_size, 150))
    with open('jester_dataset_2/jester_ratings.dat') as file:
        lines = file.readlines()
        if mode == 'random':
            sample = random.sample([i for i in range(max_user_size)], user_size)
            for line in lines:
                user_id, item_id, rating = line.split('\t\t')
                user_id = int(user_id) - 1
                if user_id in sample:
                    ratings[int(user_id) - 1, int(item_id) - 1] = round(float(rating) + 11)
        elif mode == 'top':
            for line in lines:
                user_id, item_id, rating = line.split('\t\t')
                if int(user_id) > user_size:
                    break
                ratings[int(user_id) - 1, int(item_id) - 1] = round(float(rating) + 11)
        else:
            raise ValueError
    dump(filename, ratings)
    return ratings


def main():
    parser = argparse.ArgumentParser(description='create ratings')
    parser.add_argument('-s', '--size', type=int, required=True)
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-m''--mode', default='top', choices=['top', 'random'])
    args = parser.parse_args()
    create_ratings(args.size, args.filename)


if __name__ == '__main__':
    main()
