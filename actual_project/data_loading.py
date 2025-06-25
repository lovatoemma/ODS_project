import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split

#####################
### THREE DATASET ###
#####################

# 1) MOVIELENS 100K
# Aggiunto: subset_percent_rows e subset_percent_cols per ridurre la dimensione del dataset

def load_movielens(path_to_data='data/ml-100k/u.data', subset_percent_rows=1.0, subset_percent_cols=1.0):
    """Loads and prepares the MovieLens 100k dataset. Normalizes ratings to [0, 1]. Optionally subselects rows/cols."""
    try:
        df = pd.read_csv(path_to_data, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    except FileNotFoundError:
        print("Dataset not found. Please download MovieLens 100K and place u.data in a 'ml-100k' folder.")
        print("Download from: https://grouplens.org/datasets/movielens/100k/")
        return None, None

    # Subset users
    if subset_percent_rows < 1.0:
        user_sample = np.random.choice(df['user_id'].unique(),
                                       int(len(df['user_id'].unique()) * subset_percent_rows),
                                       replace=False)
        df = df[df['user_id'].isin(user_sample)]
    # Subset items
    if subset_percent_cols < 1.0:
        item_sample = np.random.choice(df['item_id'].unique(),
                                       int(len(df['item_id'].unique()) * subset_percent_cols),
                                       replace=False)
        df = df[df['item_id'].isin(item_sample)]

    # Normalizza rating in [0, 1]
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    if max_rating > min_rating:
        df['rating'] = (df['rating'] - min_rating) / (max_rating - min_rating)
    else:
        df['rating'] = 0.0

    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    num_users, num_items = len(user_map), len(item_map)
    
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)
    
    train_matrix = csc_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['rating'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))
                                 
    return train_matrix, test_matrix


# 2) JESTER 4
# Aggiunto: subset_percent_rows e subset_percent_cols

def load_jester(path_to_data='data/jesterDataset4/jester_data.xlsx', subset_percent_rows=1.0, subset_percent_cols=1.0):
    """Loads and prepares the Jester 4 dataset from an Excel file (matrix format). Normalizes ratings to [0, 1]. Optionally subselects rows/cols."""
    try:
        df = pd.read_excel(path_to_data, header=None)
        # print(df.head())
    except FileNotFoundError:
        print("Dataset not found. Please download Jester 4 and place jester_data.xlsx in a 'jester-data' folder.")
        print("Download from: https://eigentaste.berkeley.edu/dataset/")
        return None, None
    if df.shape[1] > 100:
        df = df.iloc[:, 1:]
    ratings = df.replace(99, 0).fillna(0).values
    # Subset users
    if subset_percent_rows < 1.0:
        n_rows = int(ratings.shape[0] * subset_percent_rows)
        row_idx = np.random.choice(ratings.shape[0], n_rows, replace=False)
        ratings = ratings[row_idx, :]
    # Subset items
    if subset_percent_cols < 1.0:
        n_cols = int(ratings.shape[1] * subset_percent_cols)
        col_idx = np.random.choice(ratings.shape[1], n_cols, replace=False)
        ratings = ratings[:, col_idx]
    min_rating = ratings.min()
    max_rating = ratings.max()
    if max_rating > min_rating:
        ratings = (ratings - min_rating) / (max_rating - min_rating)
    else:
        ratings = np.zeros_like(ratings)
    ratings_sparse = csc_matrix(ratings)
    num_users = ratings_sparse.shape[0]
    idx = np.arange(num_users)
    train_idx, test_idx = train_test_split(idx, test_size=0.4, random_state=42)
    train_matrix = ratings_sparse[train_idx, :]
    test_matrix = ratings_sparse[test_idx, :]
    return train_matrix, test_matrix

# 3) STEAM
# Aggiunto: subset_percent_rows e subset_percent_cols

def load_steam(path_to_data='data/steam/game_play.dat', subset_percent_rows=1.0, subset_percent_cols=1.0):
    """
    Loads and prepares the Steam dataset from game_play.dat.
    Uses playtime as implicit rating. Normalizes playtime to [0, 1]. Optionally subselects rows/cols.
    """
    try:
        df = pd.read_csv(path_to_data, sep='\t', header=0)
        df.columns = ['User_ID', 'tGame_ID', 'tHours']
        # print(df.head())
    except FileNotFoundError:
        print("Dataset not found. Please download Steam dataset and place game_play.dat in a 'steam-data' folder.")
        print("Download from: https://www.kaggle.com/datasets/gregorut/videogamesales")
        return None, None
    df = df[df['tHours'].notna()]
    # Subset users
    if subset_percent_rows < 1.0:
        user_sample = np.random.choice(df['User_ID'].unique(),
                                       int(len(df['User_ID'].unique()) * subset_percent_rows),
                                       replace=False)
        df = df[df['User_ID'].isin(user_sample)]
    # Subset items
    if subset_percent_cols < 1.0:
        item_sample = np.random.choice(df['tGame_ID'].unique(),
                                       int(len(df['tGame_ID'].unique()) * subset_percent_cols),
                                       replace=False)
        df = df[df['tGame_ID'].isin(item_sample)]
    min_hours = df['tHours'].min()
    max_hours = df['tHours'].max()
    if max_hours > min_hours:
        df['tHours'] = (df['tHours'] - min_hours) / (max_hours - min_hours)
    else:
        df['tHours'] = 0.0
    user_map = {uid: i for i, uid in enumerate(df['User_ID'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['tGame_ID'].unique())}
    df['user_idx'] = df['User_ID'].map(user_map)
    df['item_idx'] = df['tGame_ID'].map(item_map)
    num_users, num_items = len(user_map), len(item_map)
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)
    train_matrix = csc_matrix((train_df['tHours'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['tHours'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))
    return train_matrix, test_matrix