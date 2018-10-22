import numpy as np
from rs import unknown_rating, load, dump
import argparse

def create_ratings(user_size, filename):
    ratings = np.zeros(shape=(user_size, 150))
    with open('jester_dataset_2/jester_ratings.dat') as file:
        lines = file.readlines()
        for line in lines:
            user_id, item_id, rating = line.split('\t\t')
            if int(user_id) >= user_size:
                break
            ratings[int(user_id) - 1, int(item_id) - 1] = round(float(rating) + 21)
    dump(filename, ratings)
    return ratings


def main():
    parser = argparse.ArgumentParser(description='create ratings')
    parser.add_argument('-s', '--size', type=int, required=True)
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()
    create_ratings(args.size, args.filename)
