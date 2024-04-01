#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',  None)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv("TaxiFare.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df = df.dropna(how = 'any', axis = 'rows')


# In[9]:


df.isnull().sum()


# In[10]:


sns.countplot(x="amount",data=df,palette="coolwarm")


# In[11]:


df["no_of_passenger"].value_counts()


# In[12]:


sns.countplot(x="amount",hue="no_of_passenger",data=df)


# In[13]:


pd.crosstab(df["no_of_passenger"],df["amount"])


# In[14]:


plt.figure(figsize=(10,6))
df['no_of_passenger'].value_counts().plot.bar(color='y',edgecolor='k')
plt.title('Passenger Count')
plt.xlabel('Passenger count',fontsize=15)
plt.ylabel('Count',fontsize=15)


# In[15]:


df['amount'].describe()


# In[16]:


df['amount'] = df['amount'].apply(np.int64)


# In[17]:


df.isnull().sum()


# In[18]:


X=df.drop("amount",axis=1)
Y=df["amount"]


# In[24]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=12)


# In[25]:


model_1=LogisticRegression()


# In[26]:


X=X.apply(pd.to_numeric,errors='coerce')
Y=Y.apply(pd.to_numeric,errors='coerce')


# In[27]:


X.fillna(1,inplace=True)
Y.fillna(1,inplace=True)


# In[28]:


model_1.fit(X_train,Y_train)


# In[29]:


model_1.score(X_train,Y_train)


# In[30]:


model_1.score(X_test,Y_test)


# In[31]:


predictions=model_1.predict(X_test)


# In[32]:


from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[33]:


accuracy_score(Y_test,predictions)


# In[34]:


from sklearn.tree import DecisionTreeClassifier

model_2=DecisionTreeClassifier(max_depth=3)
model_2.fit(X_train,Y_train)


# In[35]:


model_2.score(X_train,Y_train)


# In[36]:


model_2.score(X_test,Y_test)


# In[37]:


from sklearn.ensemble import BaggingClassifier
model_3=BaggingClassifier(n_estimators=90,base_estimator=model_2)
model_3.fit(X_train,Y_train)


# In[38]:


model_3.score(X_train,Y_train)


# In[39]:


model_3.score(X_test,Y_test)


# In[40]:


from sklearn.ensemble import AdaBoostClassifier
model_4=AdaBoostClassifier(n_estimators=30)
model_4.fit(X_train,Y_train)


# In[41]:


model_4.score(X_train,Y_train)


# In[42]:


model_4.score(X_test,Y_test)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
model_5=RandomForestClassifier(n_estimators=30)
model_5.fit(X_train,Y_train)


# In[46]:


model_5.score(X_train,Y_train)


# In[47]:


model_5.score(X_test,Y_test)


# In[ ]:




