# -*- coding: utf-8 -*-
"""Untitled13.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W7x-8mBq_uZ56BoVaHJBRHA-Z8msYOp5
"""
SOURCE - source - https://github.com/gbolmier/funk-svd/tree/master/funk_svd( my contribution little modificatins and additional mappings)
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
def _preprocess_data(X):

        X = X.copy()
        #X_test=X_test.copy()
 # Mappings have to be created
        #X.sort_values(by=[)
        user_ids = X['u_id'].unique().tolist()
        item_ids = X['i_id'].unique().tolist()

        n_users = len(user_ids)
        n_items = len(item_ids)

        user_idx = range(n_users)
        item_idx = range(n_items)

        user_mapping_ = dict(zip(user_ids, user_idx))
        item_mapping_ = dict(zip(item_ids, item_idx))

        X['u_id'] = X['u_id'].map(user_mapping_)
        X['i_id'] = X['i_id'].map(item_mapping_)
        #X_test['u_MAP'] = X_test['u_id'].map(user_mapping_)
        #X_test['i_MAP'] = X_test['i_id'].map(item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)
        #X_test.fillna(-1, inplace=True)
        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)
        #X_test['u_id'] = X_test['u_id'].astype(np.int32)
        #X_test['i_id'] = X_test['i_id'].astype(np.int32)
        return X[['u_id','i_id','rating']],user_mapping_,item_mapping_
