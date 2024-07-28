#!/usr/bin/env python
# coding: utf-8

# In[162]:


import pandas as pd


# In[163]:


train=pd.read_csv('fraudTrain.csv')


# In[164]:


train.head()


# In[165]:


test=pd.read_csv('fraudTest.csv')


# In[166]:


test.head()


# In[167]:


print(train.shape)


# In[168]:


test.shape


# In[169]:


train.isnull().any()


# In[170]:


test.isnull().any()


# In[171]:


print(train.isnull().sum())


# In[172]:


print(test.isnull().sum())


# In[173]:


train.info()


# In[174]:


train['is_fraud'].value_counts()


# In[175]:


df=pd.concat([train,test])
df.head()


# In[176]:


df.isnull().any()


# In[177]:


df.shape


# In[178]:


df['is_fraud'].value_counts()


# In[179]:


duplicates = df.duplicated()
duplicates.sum()


# In[ ]:





# In[180]:


df.drop_duplicates()


# In[181]:


df.shape


# In[182]:


x=df.drop('is_fraud',axis=1)
y=df['is_fraud']


# In[183]:


x.head()


# In[184]:


y.head()


# In[185]:


print(x.shape)
print(y.shape)


# In[186]:


legit=df[df.is_fraud==0]
fraud=df[df.is_fraud==1]


# In[187]:


legit


# In[188]:


fraud


# In[189]:


print(legit.shape)
print(fraud.shape)


# In[190]:


legit_samples=legit.sample(n=9651)


# In[191]:


new_df=pd.concat([legit_samples,fraud])


# In[192]:


new_df.head()


# In[193]:


new_df.shape


# In[194]:


new_df.info()


# In[195]:


new_df.groupby('is_fraud').count()['cc_num'].plot.bar()


# In[196]:


columns_dropped = ['Unnamed: 0',
                   'merchant', 
                   'cc_num',
                   'first', 
                   'last',
                   'gender',
                   'trans_num',
                   'unix_time',
                   'street',
                   'merch_lat',
                   'merch_long',
                   'job',
                   'zip',
                   ]

new_df.drop(columns = columns_dropped, inplace = True)


# In[197]:


new_df


# In[198]:


new_df['trans_date_trans_time'] = pd.to_datetime(new_df['trans_date_trans_time'])
new_df['dob'] = pd.to_datetime(new_df['dob'])


# In[199]:


new_df['trans_date_trans_time'] = new_df['trans_date_trans_time'].dt.hour
new_df= new_df.rename(columns = {'trans_date_trans_time': 'hour_transaction'})


# In[200]:


def get_tod(hour):
    if 4 < hour['hour_transaction'] <= 12:
        ans = 'morning'
    elif 12 < hour['hour_transaction'] <= 20:
        ans = 'afternoon'
    elif hour['hour_transaction'] <= 4 or hour['hour_transaction'] > 20:
        ans = 'night'
    return ans
new_df['hour_transaction'] = new_df.apply(get_tod, axis = 1)
new_df.head()


# In[201]:


new_df['dob']= new_df['dob'].dt.year
new_df = new_df.rename(columns = {'dob': 'age'})
from datetime import datetime
new_df['age'] = datetime.now().year - new_df['age']
# Analyzing how many frauds occur for each age group
new_df[new_df['is_fraud'] == 1].groupby('age').count()['is_fraud']


# In[202]:


NUMERICAL_FEATURES = [i for i in new_df.columns if new_df[i].dtype == 'int64'\
                      or new_df[i].dtype =='int32' \
                      or new_df[i].dtype =='float64']
CATEGORICAL_FEATURES = [i for i in new_df if new_df[i].dtype == 'object']


# In[203]:


NUMERICAL_FEATURES


# In[204]:


CATEGORICAL_FEATURES


# In[205]:


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
encoder.fit(new_df[CATEGORICAL_FEATURES])

new_df[CATEGORICAL_FEATURES] = encoder.transform(new_df[CATEGORICAL_FEATURES])
new_df.head()


# In[206]:


new_df[['is_fraud', 'age']] = new_df[['is_fraud', 'age']].astype('float64')


# In[207]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
new_df_scaled=sc.fit_transform(new_df)
new_df_scaled=pd.DataFrame(new_df_scaled)

last_column = new_df_scaled.shape[1]-1
new_df_scaled.rename(columns={last_column: 'is_fraud'}, inplace=True)
new_df_scaled.head()


# In[ ]:





# In[221]:


x = new_df_scaled.drop(columns = 'is_fraud')

# y = target values, last column of the data frame
y = new_df_scaled['is_fraud']
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_test.shape


# In[216]:


from sklearn.linear_model import LogisticRegression
k=LogisticRegression()
k.fit(x_train,y_train)


# In[217]:


from sklearn.metrics import accuracy_score
y_p=k.predict(x_test)
print(accuracy_score(y_test,y_p))


# In[229]:


pred = k.predict([[0.2,0.67,0.23,1,0.98,0.77,0.54,0.89,0.76]])
pred[0]


# In[230]:


if pred[0] == 0:
    print("Normal Transcation")
else:
    print("Fraud Transcation")

