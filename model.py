#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
import pickle
import re


# In[2]:


stop_words = stopwords.words('english')
stemmer = PorterStemmer()


# ###  Loading Text processing steps

# In[3]:


def clean_data(text):
    # Make the text lowercase
    text = text.lower()
    
    # Remove text in square brackets
    text = re.sub(r'\[.*?\]','',text)
    
    # Remove punctuation
    text = re.sub(r'[^A-Za-z]+',' ',text)
    
    # Remove words containing numbers    
    text = re.sub("\S*\d\S*", "", text).strip()
    
    text_formatted = ' '.join([words for words in text.split() if words not in stop_words])
    
    text_formatted_final = ' '.join([stemmer.stem(words) for words in word_tokenize(text_formatted)])
    
    return text_formatted_final


# ### Loading Vectorizer

# In[4]:


vector = pickle.load(open('Models/vector.pkl','rb'))


# ### Loading Random forest

# In[5]:


rf = pickle.load(open('Models/rf.pkl','rb'))


# ### Loading cleaned data and Recommendation model below

# In[6]:


data_fil = pickle.load(open('Models/cleaned_data.pkl','rb'))


# In[7]:


user_final_rating = pickle.load(open('Models/user_final_rating.pkl','rb'))


# ### Creating get recommendations by user function to use in Flask app

# In[8]:


def getrecommendationsbyuser(user_input):
    if user_input in user_final_rating.index:
        final = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
        df = data_fil[data_fil['name'].isin(list(final.index))][['name','reviews_text']]
        ls = []
        for sentence in df['reviews_text']:
            ls.append(clean_data(sentence))
        df['clean_review']=ls    
        df['sentiment_predicted']=rf.predict(vector.transform(df['clean_review']).toarray())
        df_5=df.groupby('name').mean('sentiment_predicted').sort_values('sentiment_predicted',ascending=False)[0:5]
        return ' ; '.join(list(df_5.index))
    else:
        return 'Not a user, enter a valid username'


# In[9]:


getrecommendationsbyuser('adam')


# In[10]:


user_final_rating.loc['alexis'].sort_values(ascending=False)[0:20]


# In[ ]:




