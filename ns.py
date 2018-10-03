from rs import *
import logging
import math

def sim_rate(rate1,rate2):
    return math.exp(-abs(rate1-rate2)/rate_scale)

def score(aux,record):
    item_ids=list(np.nonzero(aux)[0])
    sum=0
    for item_id in item_ids:
        weight=1/supp_item(item_id)
        sum+=weight*sim_rate(aux[item_id],record[item_id])
    return sum

def de_anonymization(aux,eccen):
    candidates=[]
    for i in range(len(users)):
        record=user(i)
        candidates.append(score(aux,record))
    temp=candidates.copy()
    std=np.std(temp)
    max1=max(temp)
    temp.remove(max1)
    max2=max(temp)
    if (max1-max2)/std<eccen:
        return None
    else:
        return user(candidates.index(max1))

def entropic_de(aux):
    candidates = []
    for i in range(len(users)):
        record = user(i)
        candidates.append(score(aux, record))
    std=np.std(candidates)
    temp=[i/std for i in candidates]
    total=sum(temp)
    c=1/total
    return (c,temp)


