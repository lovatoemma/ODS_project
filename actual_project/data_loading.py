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
def load_movielens(path_to_data='data/ml-100k/u.data'):
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


# 2) JESTER 4 (over 100K ratings)
def load_jester(path_to_data='data/jester-data/jester_data.xlsx'):
    """Loads and prepares the Jester 4 dataset from an Excel file."""
    try:
        df = pd.read_excel(path_to_data, header=None)
    except FileNotFoundError:
        print("Dataset not found. Please download Jester 4 and place jester_data.xlsx in a 'jester-data' folder.")
        print("Download from: https://eigentaste.berkeley.edu/dataset/")
        return None, None

    # Rimuovi eventuali colonne di ID utente se presenti (Jester a volte ha la prima colonna come count o ID)
    if df.shape[1] > 100:  # tipico: 100 items, una colonna extra
        df = df.iloc[:, 1:]

    # Trasforma in formato long: ogni riga = (user, item, rating)
    df_long = df.stack().reset_index()
    df_long.columns = ['user_id', 'item_id', 'rating']

    # Filtra solo i rating realmente presenti (Jester usa 99 o NaN per missing)
    df_long = df_long[~df_long['rating'].isna()]
    df_long = df_long[df_long['rating'] != 99]

    user_map = {uid: i for i, uid in enumerate(df_long['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df_long['item_id'].unique())}

    df_long['user_idx'] = df_long['user_id'].map(user_map)
    df_long['item_idx'] = df_long['item_id'].map(item_map)

    num_users, num_items = len(user_map), len(item_map)

    train_df, test_df = train_test_split(df_long, test_size=0.2, random_state=42)

    train_matrix = csc_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['rating'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))

    return train_matrix, test_matrix

# 3) STEAM (200K ratings)
def load_steam(path_to_data='steam-data/game_play.dat'):
    """
    Loads and prepares the Steam dataset from game_play.dat.
    Uses playtime as implicit rating.
    """
    try:
        # Il file è separato da virgole, senza header: user_id, item_id, playtime
        df = pd.read_csv(path_to_data, header=None, names=['user_id', 'item_id', 'playtime'])
    except FileNotFoundError:
        print("Dataset not found. Please download Steam dataset and place game_play.dat in a 'steam-data' folder.")
        print("Download from: https://www.kaggle.com/datasets/gregorut/videogamesales")
        return None, None

    # Filtra eventuali righe senza playtime valido
    df = df[df['playtime'].notna()]
    # Puoi anche filtrare playtime=0 se vuoi solo interazioni "positive"
    # df = df[df['playtime'] > 0]

    # Mappa user e item a indici consecutivi
    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}

    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)

    num_users, num_items = len(user_map), len(item_map)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_matrix = csc_matrix((train_df['playtime'], (train_df['user_idx'], train_df['item_idx'])), shape=(num_users, num_items))
    test_matrix = csc_matrix((test_df['playtime'], (test_df['user_idx'], test_df['item_idx'])), shape=(num_users, num_items))

    return train_matrix, test_matrix