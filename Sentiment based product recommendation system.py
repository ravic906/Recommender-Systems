#!/usr/bin/env python
# coding: utf-8

# ### Importing packages

# In[1]:


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle


# ### Reading data

# In[2]:


data = pd.read_csv('sample30.csv')
data.head()


# ### Checking nulls

# In[3]:


data.isnull().sum()*100/data.shape[0]


# In[4]:


data.drop(['reviews_didPurchase','reviews_userCity','reviews_userProvince'],1,inplace=True)


# ### Dropped fields that have more than 30% of nulls 

# In[5]:


data.isnull().sum()*100/data.shape[0]


# ### Checking data types and aligning them

# In[6]:


data.info()


# In[7]:


data.head()


# In[8]:


data.reviews_doRecommend.value_counts()


# In[9]:


data_fil = data[data['reviews_doRecommend'].isnull()!=True]


# In[10]:


data_fil = data_fil[data_fil['user_sentiment'].isnull()!=True]


# In[11]:


data_fil.info()


# In[12]:


data_fil.head()


# ### Dropping the ID column as it is not required

# In[13]:


data_fil.drop('id',1,inplace=True)


# In[14]:


data_fil['user_sentiment'] = data_fil['user_sentiment'].map({'Positive':1,'Negative':0})


# In[15]:


data_fil.brand.value_counts()


# In[16]:


data_fil['reviews_date']= data_fil['reviews_date'].apply(lambda x: x[:-14])


# In[17]:


data_fil['reviews_date'].value_counts()


# In[18]:


data_fil = data_fil[data_fil['reviews_date']!=' hooks slide or swivel into any desi']


# In[19]:


data_fil['reviews_date'] = pd.to_datetime(data_fil['reviews_date'], format='%Y-%m-%d')


# In[20]:


data_fil.head()


# In[21]:


data_fil = data_fil[data_fil['reviews_username'].isna()!=True]


# In[22]:


data_fil.reviews_doRecommend = data_fil.reviews_doRecommend.astype(bool)


# In[23]:


data_fil.info()


# ### Text processing

# In[24]:


def clean_data(text):
    # Make the text lowercase
    text = text.lower()
    
    # Remove text in square brackets
    text = re.sub(r'\[.*?\]','',text)
    
    # Remove punctuation
    text = re.sub(r'[^A-Za-z]+',' ',text)
    
    # Remove words containing numbers    
    text = re.sub("\S*\d\S*", "", text).strip()
    
    return text


# In[25]:


stop_words = stopwords.words('english')


# In[26]:


def stp(sentence):
    return ' '.join([words for words in sentence if words not in stop_words])


# In[27]:


stemmer = PorterStemmer()


# In[28]:


def stem(sentence):
    return ' '.join([stemmer.stem(words) for words in sentence])


# In[29]:


data_fil['reviews_text'] = data_fil['reviews_text'].apply(lambda x:clean_data(x))


# In[30]:


data_fil['reviews_text'] =data_fil['reviews_text'].apply(lambda x:stp(x.split()))


# In[31]:


data_fil['reviews_text'] = data_fil['reviews_text'].apply(lambda x: stem(word_tokenize(x)))


# In[32]:


data_fil['reviews_text']


# In[33]:


data_fil['user_sentiment'].value_counts(normalize=True)


# ### There is class imbalance, need to balance it

# ### TFIDF - Creating a vectorizer

# In[34]:


vector = TfidfVectorizer()


# In[35]:


cool = vector.fit_transform(data_fil['reviews_text'])


# In[36]:


tfidf = pd.DataFrame(cool.toarray(),columns=vector.get_feature_names_out())


# ### Dumping the TFIDF as pickle

# In[37]:


pickle.dump(vector,open('Models/vector.pkl','wb'))


# ### Using SMOTE to balance the data

# In[38]:


X=tfidf
y=data_fil['user_sentiment']


# In[39]:


X.shape


# In[40]:


y.shape


# In[41]:


smote = SMOTE(random_state=1)
X_smote, y_smote = smote.fit_resample(X,y)


# In[42]:


y_smote.value_counts(normalize=True)


# ### Train test split

# In[43]:


X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,random_state=1,train_size=0.7)


# In[44]:


X_train.shape


# In[45]:


X_test.shape


# In[46]:


y_train.shape


# In[47]:


y_test.shape


# In[48]:


from sklearn.metrics import f1_score,precision_score,recall_score


# ### Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)


# In[50]:


y_test_pred = logreg.fit(X_train,y_train).predict(X_test)


# In[51]:


f1_score(y_test_pred,y_test,average='weighted')


# In[52]:


recall_score(y_test_pred,y_test,average='weighted')


# In[53]:


precision_score(y_test_pred,y_test,average='weighted')


# ### Decision Tree

# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


dt = DecisionTreeClassifier()


# In[56]:


y_test_pred_dt = dt.fit(X_train,y_train).predict(X_test)


# In[57]:


f1_score(y_test_pred_dt,y_test,average='weighted')


# In[58]:


recall_score(y_test_pred_dt,y_test,average='weighted')


# In[59]:


precision_score(y_test_pred_dt,y_test,average='weighted')


# ### Random Forest

# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


rf = RandomForestClassifier()


# In[62]:


y_test_pred_rf = rf.fit(X_train,y_train).predict(X_test)


# In[63]:


f1_score(y_test_pred_rf,y_test,average='weighted')


# In[64]:


recall_score(y_test_pred_rf,y_test,average='weighted')


# In[65]:


precision_score(y_test_pred_rf,y_test,average='weighted')


# ### As scores are much larger for Random forest, we are going to use it for prediction

# ### Saving the prediction model

# In[66]:


pickle.dump(rf,open('Models/rf.pkl','wb'))


# ## Starting on recommendation process

# In[67]:


# Reading ratings file from GitHub. # MovieLens
ratings = pd.read_csv('sample30.csv')
ratings.head()


# ## Dividing the dataset into train and test

# In[68]:


# Test and Train split of the dataset.
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size=0.30, random_state=31)


# In[69]:


print(train.shape)
print(test.shape)


# In[70]:


# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are usernames.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)

df_pivot.head(5)


# In[71]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()


# In[72]:


# The movies not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[73]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
      index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)


# In[74]:


dummy_train.head()


# # User Similarity Matrix

# ## Using Cosine Similarity

# In[75]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[76]:


user_correlation.shape


# ## Using adjusted Cosine 

# ### Here, we are not removing the NaN values and calculating the mean only for the products rated by the user

# In[77]:


# Create a user-movie matrix.
df_pivot = train.pivot_table(
     index='reviews_username',
    columns='name',
    values='reviews_rating'
)


# In[78]:


df_pivot.head()


# ### Normalising the rating of the product for each user around 0 mean

# In[79]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[80]:


df_subtracted.head()


# ### Finding cosine similarity

# In[81]:


from sklearn.metrics.pairwise import pairwise_distances


# In[82]:


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# ## Prediction - User User

# In[83]:


user_correlation[user_correlation<0]=0
user_correlation


# In[84]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[85]:


user_predicted_ratings.shape


# In[86]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### Finding the top 5 recommendation for the *user*

# In[87]:


# Take the user ID as input.
user_input = "alexis"
print(user_input)


# In[88]:


user_final_rating.head(2)


# In[89]:


d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# In[90]:


#Mapping with Movie Title / Genres 
product_mapping = pd.read_csv('sample30.csv')
product_mapping.head()


# In[91]:


d = pd.merge(d,product_mapping,left_on='name',right_on='name', how = 'left')
d.head()


# # Evaluation - User User 

# In[92]:


# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape


# In[93]:


common.head()


# In[94]:


# convert into the user-movie matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[95]:


# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)


# In[96]:


df_subtracted.head(1)


# In[97]:


user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[98]:


common.head(1)


# In[99]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[100]:


user_correlation_df_1.shape


# In[101]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[102]:


user_correlation_df_3 = user_correlation_df_2.T


# In[103]:


user_correlation_df_3.head()


# In[104]:


user_correlation_df_3.shape


# In[105]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[106]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)


# In[107]:


dummy_test.shape


# In[108]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[109]:


common_user_predicted_ratings.head(2)


# Calculating the RMSE for only the movies rated by user. For RMSE, normalising the rating to (1,5) range.

# In[110]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[111]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[112]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[113]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# In[114]:


common_.head()


# ## Using Item similarity

# # Item Based Similarity

# In[115]:


df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).T

df_pivot.head()


# In[116]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[117]:


df_subtracted.head()


# In[118]:


from sklearn.metrics.pairwise import pairwise_distances


# In[119]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[120]:


item_correlation[item_correlation<0]=0
item_correlation


# # Prediction - Item Item

# In[121]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[122]:


item_predicted_ratings.shape


# In[123]:


dummy_train.shape


# ### Filtering the rating only for the products not rated by the user for recommendation

# In[124]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# ### Finding the top 5 recommendation for the *user*
# 
# 

# In[125]:


# Take the user ID as input
user_input = "alexis"
print(user_input)


# In[126]:


# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# In[127]:


#Mapping with Movie Title / Genres 
prod_mapping = pd.read_csv('sample30.csv')


# In[128]:


d = pd.merge(d,prod_mapping,left_on='name',right_on='name',how = 'left')
d.head()


# In[129]:


#train_new = pd.merge(train,prod_mapping,left_on='name',right_on='name',how='left')
#train_new[train_new.reviews_username_x == 'wkbu'].head()


# # Evaluation - Item Item

# In[130]:


test.columns


# In[131]:


common =  test[test.name.isin(train.name)]
common.shape


# In[132]:


common.head(4)


# In[133]:


common_item_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T


# In[134]:


common_item_based_matrix.shape


# In[135]:


item_correlation_df = pd.DataFrame(item_correlation)


# In[136]:


item_correlation_df.head(1)


# In[137]:


item_correlation_df['name'] = df_subtracted.index
item_correlation_df.set_index('name',inplace=True)
item_correlation_df.head()


# In[138]:


list_name = common.name.tolist()


# In[139]:


item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[140]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T


# In[141]:


item_correlation_df_3.head()


# In[142]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[143]:


common_item_predicted_ratings.shape


# In[144]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.
# 

# In[145]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T


# In[146]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[147]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[148]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### As there is less error on User based recommendation system, we are going to use it further

# In[149]:


#Saving the cleaned_data to a pickle
pickle.dump(data_fil,open('Models/cleaned_data.pkl','wb'))


# In[150]:


#Saving the user based recommendation to a pickle
pickle.dump(user_final_rating,open('Models/user_final_rating.pkl','wb'))


# ### Fine-Tuning the Recommendation System and Recommendation of Top 5 Products

# In[151]:


def getrecommendationsbyuser(user_input):
#user_input = input("Enter your user name")
#print(user_input)
    final = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    df = data_fil[data_fil['name'].isin(list(final.index))][['name','reviews_text']]
    df['sentiment_predicted']=rf.predict(vector.transform(df['reviews_text']).toarray())
    df_5=df.groupby('name').mean('sentiment_predicted').sort_values('sentiment_predicted',ascending=False)[0:5]
    return list(df_5.index)


# In[152]:


getrecommendationsbyuser('alex')


# In[153]:


getrecommendationsbyuser('adam')


# In[154]:


getrecommendationsbyuser('alexis')


# In[155]:


user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]


# In[ ]:




