#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
!pip install -U -q git+https://github.com/sparsh-ai/recochef.git
import numpy as np
import matplotlib.pyplot as plt
import pandas
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.optim.lr_scheduler import _LRScheduler
from recochef.utils.iterators import batch_generator
import logging
import math
import copy
import pickle
import numpy as np
import pandas as pd
from textwrap import wrap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
import time
import logging
import argparse
import datetime

# In[2]:


def _preprocess_data(X):
        #SOURCE - source - https://github.com/gbolmier/funk-svd/tree/master/funk_svd( my contribution little modificatins and additional mappings)

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


# In[3]:


from recochef.utils.iterators import batch_generator
from recochef.models.embedding import EmbeddingNet
class MLP(nn.Module):
    def __init__(self, embedding_size, hidden_size,num_users,num_items):
        super(MLP, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.item_embeddings = nn.Embedding(num_items, embedding_size)
        self.Rec = nn.Sequential(
                      nn.Linear(embedding_size*2, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Linear(hidden_size, 1))

    def forward(self,user_id,item_id):
        self.user_embedding = self.user_embeddings(user_id)
        self.item_embedding = self.item_embeddings(item_id)
        x = torch.cat([self.user_embedding, self.item_embedding], dim=1)
        x = self.Rec(x)
        return x


# In[4]:


def train(X_train,y_train,batch,model,loss_fn,optimizer):
  running_loss=0.0
  for X_batch,y_batch in batch_generator(X_train,y_train, shuffle=True, bs=batch):
      model.train()
      prediction = model(X_batch[:, 0], X_batch[:, 1])
      loss = loss_fn(prediction, y_batch)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      running_loss+=loss.item()
  epoch_loss = running_loss / len(X_train)
  return epoch_loss 
def validation(X_valid,y_valid,batch,model,loss_fn,optimizer):
  running_loss=0.0
  for X_batch,y_batch in batch_generator(X_valid,y_valid, shuffle=True, bs=batch):
      model.eval()
      prediction = model(X_batch[:, 0], X_batch[:, 1])
      loss = loss_fn(prediction, y_batch)
      running_loss += loss.item()
  epoch_loss = running_loss/ len(X_valid)
  return epoch_loss


# In[18]:


def run_epochs(features,hidden_sz,learning_rate,batch,epochs,patience,training_data,testing_data):
    X_train = training_data[['u_id','i_id']]
    y_train = training_data[['rating']]
    X_valid = testing_data[['u_id','i_id']]
    y_valid = testing_data[['rating']]
    datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
    dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}
    n_users = training_data.u_id.nunique()
    n_movies = training_data.i_id.nunique()
    trainloss=[]
    valloss=[]
    model1 = MLP(embedding_size=features, hidden_size=hidden_sz,num_users=n_users,num_items=n_movies)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    epochs=epochs
    batch=batch
    patience = patience
    no_improvements = 0
    best_loss = np.inf
    log_file1 = os.path.join(os.getcwd(), 'metrics.log')
    logging.basicConfig(filename=log_file1, level=logging.INFO,filemode='w')
    current_time = datetime.datetime.now()
    logging.info('Current date and time: {}'.format(current_time))
    logging.info('Start')
    for i in range(epochs):
        start = time.time()
        train_loss=train(X_train,y_train,batch,model1,loss_fn,optimizer)
        trainloss.append(train_loss)
        val_loss=validation(X_valid,y_valid,batch,model1,loss_fn,optimizer)
        valloss.append(val_loss)
        end = time.time()
        logging.info('Epoch %d ;train_MSE = %.4f ; val_MSE = %.4f ; time = %.4f', i+1,train_loss,val_loss,(end-start))
        if val_loss < best_loss:
          best_loss = val_loss
        else:
          no_improvements += 1
        if no_improvements >= patience:
          break
    logging.info('End')
    return trainloss,valloss


# In[19]:


def Rec_Model(trainpath,testpath,features,hidden_sz,learning_rate,batch,epochs,patience):
  df=pd.read_csv(trainpath,sep='\t',compression='gzip',encoding='latin-1',header=None)
  df_test=pd.read_csv(testpath,sep='\t',compression='gzip',encoding='latin-1',header=None)
  df.rename(columns={0: 'u_id', 1: 'i_id',3:'rating'},inplace=True)
  df_test.rename(columns={0: 'u_id', 1: 'i_id',3:'rating'},inplace=True)
  df_test.drop(2,axis='columns',inplace=True)
  df.drop(2,axis='columns',inplace=True)
  ratings_df,user_map,item_map=_preprocess_data(df)
  df_test['u_id']=df_test['u_id'].map(user_map)
  df_test['i_id']=df_test['i_id'].map(item_map)
  trainloss,valloss=run_epochs(features=features,hidden_sz=hidden_sz,learning_rate=learning_rate,batch=batch,
                               epochs=epochs,patience=patience,
                              training_data=ratings_df,testing_data=df_test)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a recommendation model')
    parser.add_argument('--train', dest='trainpath', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--test', dest='testpath', type=str, required=True, help='Path to the testing dataset')
    parser.add_argument('--features', dest='features', type=int, default=16, help='Number of features to use in the model')
    parser.add_argument('--hidden', dest='hidden_sz', type=int, default=64, help='Size of the hidden layer in the model')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', dest='batch', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--patience', dest='patience', type=int, default=3, help='Number of epochs to wait for improvement before early stopping')
    args = parser.parse_args()
    Rec_Model(args.trainpath, args.testpath, args.features, args.hidden_sz, args.learning_rate, 
              args.batch, args.epochs, args.patience)


# In[20]:





# In[ ]:




