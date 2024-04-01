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


df=pd.read_csv("h1n1_vaccine_prediction.csv")


# In[41]:


df.head(20)


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[40]:


sns.countplot(x="h1n1_vaccine",hue="no_of_children",data=df)


# In[42]:


sns.countplot(x="h1n1_vaccine",hue="no_of_adults",data=df)


# In[45]:


pd.crosstab(df["h1n1_vaccine"],df["no_of_adults"])


# In[47]:


pd.crosstab(df["no_of_children"],df["h1n1_vaccine"])


# In[8]:


X=df.drop("h1n1_vaccine",axis=1)
Y=df["h1n1_vaccine"]


# In[9]:


df['h1n1_vaccine'] = df['h1n1_vaccine'].apply(np.int64)


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.40,random_state=1)


# In[16]:


model_1=LogisticRegression()


# In[17]:


X=X.apply(pd.to_numeric,errors='coerce')
Y=Y.apply(pd.to_numeric,errors='coerce')


# In[18]:


X.fillna(1,inplace=True)
Y.fillna(1,inplace=True)


# In[19]:


model_1.fit(X_train,Y_train)


# In[20]:


model_1.score(X_train,Y_train)


# In[21]:


model_1.score(X_test,Y_test)


# In[22]:


predictions=model_1.predict(X_test)


# In[23]:


from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[24]:


accuracy_score(Y_test,predictions)


# In[25]:


cm=metrics.confusion_matrix(Y_test,predictions,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]])
columns=[i for i in ["Predict 1","Predict 0"]]
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[26]:


from sklearn.tree import DecisionTreeClassifier

model_2=DecisionTreeClassifier(max_depth=3)
model_2.fit(X_train,Y_train)


# In[27]:


model_2.score(X_train,Y_train)


# In[28]:


model_2.score(X_test,Y_test)


# In[29]:


from sklearn.ensemble import RandomForestClassifier
model_3=RandomForestClassifier(n_estimators=30)
model_3.fit(X_train,Y_train)


# In[31]:


model_3.score(X_train,Y_train)


# In[32]:


model_3.score(X_test,Y_test)


# In[33]:


from sklearn.ensemble import GradientBoostingClassifier
model_4=GradientBoostingClassifier(n_estimators=30)
model_4.fit(X_train,Y_train)


# In[34]:


model_4.score(X_train,Y_train)


# In[35]:


model_4.score(X_test,Y_test)


# In[36]:


from sklearn.ensemble import AdaBoostClassifier
model_5=AdaBoostClassifier(n_estimators=30)
model_5.fit(X_train,Y_train)


# In[37]:


model_5.score(X_train,Y_train)


# In[38]:


model_5.score(X_test,Y_test)


# In[ ]:




