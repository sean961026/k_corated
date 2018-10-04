from rs import pd_rating,directory,users
import numpy as np
import math
import sys
import logging

input_orginal_ratings_file=sys.argv[1]
default_neighbors=[i for i in range(len(users))]
default_predict_fun=pd_rating
input_mode=sys.argv[2]


def RMSE(test_set_file,predict_fun,**kwargs):
    test_set=np.loadtxt(directory+test_set_file,delimiter='\t')
    size=test_set.shape[0]
    total=0
    for i in range(size):
        record=test_set[i,:]
        test_user=record[0]
        test_item=record[1]
        kwargs.update({'user_id':test_user,'item_id':test_item})
        if 'neighbor_ids' not in kwargs:
            kwargs.update({'neighbor_ids': default_neighbors})
        test_rating=record[2]
        predicted_rating=predict_fun(**kwargs)
        total+=(test_rating-predicted_rating)**2
    return math.sqrt(total/size)

def main():
    original_ratings=np.loadtxt(input_orginal_ratings_file+'_ratings.csv',delimiter=',')

if __name__ == '__main__':
    main()
