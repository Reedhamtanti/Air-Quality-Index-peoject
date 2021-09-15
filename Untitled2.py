#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas matplotlib numpy plotly scikit-learn ')


# In[3]:


import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[4]:


air_df = pd.read_csv('city_day.csv')


# In[5]:


air_df


# In[6]:


air_df.info()


# In[7]:


air_df.isna().sum()


# In[8]:


air_df.drop(["City", "PM10", "Date","NO","NOx","NH3", "Benzene", "Toluene", "Xylene","AQI_Bucket"], axis=1, inplace=True)


# In[9]:


air_df.isna().sum()


# In[10]:


air_df = air_df.dropna().reset_index(drop=True)
air_df


# In[11]:


air_df.head()


# In[12]:


air_df.corr()


# In[15]:


plt.figure(figsize=(15,15))

for i in range(1,air_df.shape[1]+1):
    plt.subplot(4,2,i)
    sns.histplot(data=air_df[air_df.columns[i-1]],kde=True)


# In[16]:


from scipy import stats
from scipy.stats import norm


# In[17]:


columns = air_df.columns.to_list()

def QQplot(air_df,variable):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(air_df[variable],kde=True)
    plt.xlabel(variable)
    plt.subplot(1,2,2)
    stats.probplot(air_df[variable],dist='norm',plot=plt,rvalue=True)
    plt.xlabel(variable)
for i in columns:
    QQplot(air_df,i)


# In[18]:


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[19]:


X = air_df
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


# In[20]:


vif


# In[21]:


plt.figure(figsize=(15,15))

for i in range(1,air_df.shape[1]):
    plt.subplot(4,2,i)
    sns.scatterplot(x=air_df[air_df.columns[i-1]], y = air_df.AQI)


# In[26]:


plt.figure(figsize=(20,20))
for i in range(1,X.shape[1]):
    plt.subplot(4,2,i)
    sns.boxplot(x=air_df[air_df.columns[i-1]])


# In[ ]:





# In[ ]:




