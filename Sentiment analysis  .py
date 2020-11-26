#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir('C:\\Users\\mayur\\Documents')


# In[3]:


import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt


# In[4]:


data=pd.read_csv("reviews.csv")


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


text=data['comment']


# In[9]:


nltk.download('vader_lexicon')


# In[10]:


sia=SIA()


# In[11]:


results=[]
for line in text:
        pol_score=sia.polarity_scores(line)
        pol_score['text']=line
        results.append(pol_score)


# In[12]:


df=pd.DataFrame.from_records(results)


# In[13]:


df


# In[14]:


df['label']=0
df.loc[df['compound']>0,'label']=1
df.loc[df['compound']<0,'label']=-1


# In[15]:


df


# In[16]:


print(df.label.value_counts())
print(df.label.value_counts(normalize=True)*100)


# In[17]:


import seaborn as sns


# In[18]:


fig,ax=plt.subplots(figsize=(8,8))
counts=df.label.value_counts()
sns.barplot(x=counts.index,y=counts,ax=ax)
ax.set_xticklabels(['Negetive','Neutral','Positive'])
ax.set_ylabel("counts")
plt.show()


# In[19]:


fig,ax=plt.subplots(figsize=(8,8))
counts=df.label.value_counts(normalize=True)*100
sns.barplot(x=counts.index,y=counts,ax=ax)
ax.set_xticklabels(['Negetive','Neutral','Positive'])
ax.set_ylabel("counts")
plt.show()


# In[ ]:




