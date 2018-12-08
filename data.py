import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def getrating(fname, num_user, num_movie):
    df = pd.read_csv(fname,delimiter='::',names=['userID', 'movieID', 'rating', 'timestamp'],engine='python')

    num_df = len(df)
    idx_df = range(num_df)

    # train_idx, text_idx = train_test_split(idx_df, test_size=0.1, random_state=42)
    train_idx, test_idx = train_test_split(df, test_size=0.1, random_state=42)

    num_train_data = len(train_idx)
    num_test_data = len(test_idx)

    # num_user = df.userID.unique().shape[0]
    # num_movie = df.movieID.unique().shape[0]
    print('Number of user = ' + str(num_user) + '| Number of movie = ' + str(num_movie))

    data = {
        'all':{
            'r' : np.zeros((num_user,num_movie)),
            'mask': np.zeros((num_user,num_movie)),
            'user': set(),
            'movie': set(),
        },
        'train':{
            'r': np.zeros((num_user,num_movie)),
            'mask': np.zeros((num_user,num_movie)),
            'user': set(),
            'movie': set(),
        },
        'test':{
            'r': np.zeros((num_user,num_movie)),
            'mask': np.zeros((num_user,num_movie)),
            'user': set(),
            'movie': set(),
        },
    }

    for line in df.itertuples():
        data['all']['r'][line.userID-1, line.movieID-1] = line.rating
        data['all']['mask'][line.userID-1, line.movieID-1] = 1
        data['all']['user'].add(line.userID)
        data['all']['movie'].add(line.movieID)

    for line in train_idx.itertuples():
        data['train']['r'][line.userID-1, line.movieID-1] = line.rating
        data['train']['mask'][line.userID-1, line.movieID-1] = 1
        data['train']['user'].add(line.userID)
        data['train']['movie'].add(line.movieID)

    for line in test_idx.itertuples():
        data['test']['r'][line.userID-1, line.movieID-1] = line.rating
        data['test']['mask'][line.userID-1, line.movieID-1] = 1
        data['test']['user'].add(line.userID)
        data['test']['movie'].add(line.movieID)

    #print(data)
    return data,num_train_data,num_test_data


# if __name__=='__main__':
#     getrating('ml-1m/ratings.csv',6040,3952)