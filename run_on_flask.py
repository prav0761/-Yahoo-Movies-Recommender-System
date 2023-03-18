#!/usr/bin/env python
# coding: utf-8

# In[16]:


from Run_MLP_Rec import MLP
import torch
import pandas as pd

# In[2]:



# In[60]:


model = torch.load('trained_model.pth')
user_map = pd.read_pickle('user_map.pkl')
item_map = pd.read_pickle('item_map.pkl')


# In[67]:




# In[73]:


from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/predict_rating_api', methods=['POST'])
def predict():
    # get user_id and item_id from request data
    user_id = request.json['user_id']
    item_id = request.json['item_id']
    print(user_id)
    # call the predict_rating function to get the predicted rating
    rating = predict_rating(user_id, item_id, user_map, item_map, model)
    
    # return the predicted rating as a JSON response
    return jsonify({'rating': rating})

def predict_rating(user_id, item_id,user_map,item_map,model):
    # Map user and item IDs to the corresponding indices
    user_idx = user_map[user_id] if user_id in user_map else -1
    item_idx = item_map[item_id] if item_id in item_map else -1
    # If either user or item is unknown, return -1
    if user_idx == -1 or item_idx == -1:
        return -1

    # Create a tensor from the user and item indices and feed it to the model to get the predicted rating
    user_tensor = torch.tensor(user_idx, dtype=torch.long).view(1)
    item_tensor = torch.tensor(item_idx, dtype=torch.long).view(1)
    prediction = model(user_tensor, item_tensor)

    # Return the predicted rating as a float
    if prediction.item()>=5:
        return 5
    elif prediction.item()<=0:
        return 0
    else:
        return prediction.item()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9880, debug=True)



# In[ ]:




