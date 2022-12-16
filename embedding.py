#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

#get_ipython().system('pip install -U sentence-transformers')


# In[2]:


from sentence_transformers import SentenceTransformer


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from transformers import AutoTokenizer, AutoModel
import torch


# In[5]:


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# In[6]:


tokenizer = AutoTokenizer.from_pretrained('ddobokki/electra-small-nli-sts')
model = AutoModel.from_pretrained('ddobokki/electra-small-nli-sts') 


# In[7]:


# isbn 기준 -> 다른키 기준으로 하려면 밑에 3개의 isbn 을 column 이름으로 대체


# In[38]:


try:
    df = pd.read_csv(sys.argv[1],index_col=None,encoding='euc-kr',low_memory=False)
except:
    print("pd.read_csv failed. Check file name and file format")


# In[56]:


df['main_keywords'] = df['main_keywords'].fillna('') # 결측값을 빈값으로 바꾼다 결측값이 있으면 electra embedding error
df['title'] = df['title'].fillna('')


# In[ ]:


# 추가하고 싶은 column name 밑에 열에 추가하고 for loop 의 2번째 줄 list에 row['column name'] 추가


# In[61]:


df_embeddings = pd.DataFrame(columns = ['isbn','title','embedding'])



# In[62]:

print("embedding start....")

def embedd_sentences(sentences):
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt') 
  with torch.no_grad():
    model_output = model(**encoded_input)
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  embedding = np.sum(np.array(sentence_embeddings),axis=0)
  return embedding.astype('float32').tolist()


# In[63]:


for index, row in df_2.iterrows():
  df_embeddings.loc[index] = [row['isbn'],row['title'],embedd_sentences([row['title'],row['main_keywords']])]
  if index % 20000 == 0:
    print(str(index) + "...")


# In[1]:



# In[33]:


# 저장하고 싶은 파일명


# In[ ]:
print("to_csv....")

try:
    df_embeddings.to_csv(sys.argv[2],index=False)
except:
    print("pd.to_csv failed.")


# In[ ]:




