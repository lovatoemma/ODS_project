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
def load_movielens_100k(path_to_data='ml-100k/u.data'):
    """Loads and prepares the MovieLens 100k dataset."""
    try:
        df = pd.read_csv(path_to_data, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    except FileNotFoundError:
        print("Dataset not found. Please download MovieLens 100K and place u.data in a 'ml-100k' folder.")
        print("Download from: https://grouplens.org/datasets/movielens/100k/")
        return None, None

    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    num_users, num_items = len(user_map), len(item_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_matrix = csc_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['rating'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))
                                     
    return train_matrix, test_matrix

# Load the data
train_data, test_data = load_movielens_100k()


# 2) JESTER 4 (over 100K ratings)
def load_jester_4(path_to_data='jester-data/jester-data-4.csv'):
    """Loads and prepares the Jester 4 dataset."""
    try:
        df = pd.read_csv(path_to_data, header=None, names=['user_id', 'item_id', 'rating'])
    except FileNotFoundError:
        print("Dataset not found. Please download Jester 4 and place jester-data-4.csv in a 'jester-data' folder.")
        print("Download from: https://eigentaste.berkeley.edu/dataset/")
        return None, None

    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    num_users, num_items = len(user_map), len(item_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_matrix = csc_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['rating'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))
                                     
    return train_matrix, test_matrix

# 3) STEAM (200K ratings)
def load_steam(path_to_data='steam-data/steam-200k.csv'):
    """Loads and prepares the Steam dataset."""
    try:
        df = pd.read_csv(path_to_data, header=None, names=['user_id', 'item_id', 'rating'])
    except FileNotFoundError:
        print("Dataset not found. Please download Steam dataset and place steam-200k.csv in a 'steam-data' folder.")
        print("Download from: https://www.kaggle.com/datasets/gregorut/videogamesales")
        return None, None

    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    num_users, num_items = len(user_map), len(item_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_matrix = csc_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['rating'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))
                                     
    return train_matrix, test_matrix