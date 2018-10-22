import numpy as np
from rs import unknown_rating, load, dump


def create_ratings(user_size, type, filename):
    ratings = np.zeros(shape=(user_size, 150))
    with open('jester_dataset_2/jester_ratings.dat') as file:
        lines = file.readlines()
        for line in lines:
            user_id, item_id, rating = line.split('\t\t')
            if user_id >= user_size:
                break
            ratings[int(user_id) - 1, int(item_id) - 1] = type(rating)
    dump(filename, ratings)
    return ratings


create_ratings(63978, float, 'test')
