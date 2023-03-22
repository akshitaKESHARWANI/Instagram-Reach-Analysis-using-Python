#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv('C:/Users/A5327/Desktop/New folder/Python/Python/project/Instagram data.csv', encoding = 'latin1')
print(data.head())


# In[9]:


data.isnull().sum()  #contains no nul values
data=data.dropna()   #drop null values if any


# In[10]:


data.info()


# In[17]:


plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title('Distribution of Impression from home')
sns.distplot(data['From Home'])
plt.show


# In[19]:


plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title('Distribution of Impression from Hastags')
sns.distplot(data['From Hashtags'])
plt.show


# In[21]:


plt.figure(figsize=(10,8))
#plt.style.use('fivethirtyeight')
plt.title('Distribution of Impression from Explore')
sns.distplot(data['From Explore'])
plt.show


# In[30]:


home=data["From Home"].sum()
hashtags=data["From Hashtags"].sum()
explore=data["From Explore"].sum()
other=data["From Other"].sum()

labels=['From Home','From Hashtags','From Explore','Other']
values= [home, hashtags, explore, other]

fig=px.pie(data, values=values, names=labels, title='pie chart')
fig.show()


# In[34]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[45]:


text=" ".join(i for i in data.Caption)
stopwords=set(STOPWORDS)
wordcloud= WordCloud(stopwords = stopwords ,background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[46]:


text=" ".join(i for i in data.Hashtags)
stopwords=set(STOPWORDS)
wordcloud= WordCloud(stopwords = stopwords ,background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

