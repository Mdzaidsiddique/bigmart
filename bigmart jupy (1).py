#!/usr/bin/env python
# coding: utf-8

# EDA and Feature Engineering for bigmart dataset

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# importing dataset

# In[2]:


bigmart = pd.read_csv(r"C:\Users\masoo\Downloads\Bigmart.csv")


# In[3]:


bigmart.head()


# In[4]:


bigmart.shape


# Observations:
# This shows that the bigmart data is imported successfully and it consists of 8,523 rows with 12 columns.

# In[5]:


bigmart[bigmart.Item_Identifier=='FDP10']


# In[6]:


bigmart.columns


# In[7]:


#Finding some basic information about the features of the data.
bigmart.info()


# Numerical Features: Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales(Target Variable)
# 
# Categorical Features: Item_Identifier, Item_Fat_Content(Ordinal Feature), Item_Type, Outlet_Itemtifier, Outlet_Establishment_Year, Outlet_Size(Ordinal Feature), Outlet__Location_Type(Ordinal Feature), Ootlet_Type(Ordinal Feature)
# 
# Observations:
# There are 4 float type variables, 1 integer type and 7 object type.
# We are considering Item_Establishment_Year as a categorical feature because it contains some fixed value but not converting its data type now will consider later.
# Item_Fat_Content, Outlet_Size, Outelet_Location_Type and Outlet_Type are ordinal features because these values can be arranged in some order.

# In[8]:


#description about the data
bigmart.describe()


# In[9]:


#checking null values
bigmart.isnull().sum()


# In[10]:


#percentage of null values having columns
bigmart.isnull().sum()/bigmart.shape[0]*100


# Observation: we can see that Item_Identifier having 17% of null values and Outlet_Size haveing 28% of null values.

# In[11]:


#imputing null values
bigmart.Item_Weight.fillna(bigmart.Item_Weight.mean(),inplace = True)
bigmart.Outlet_Size.fillna('Not Defined',inplace = True)


# In[12]:


#droping unwanted columns
bigmart.drop(['Item_Identifier','Outlet_Identifier'], axis = 1, inplace = True)


# In[13]:


#checking Outlier
bigmart.boxplot()
plt.boxplot(bigmart.Item_Outlet_Sales )
plt.boxplot(bigmart.Item_Visibility)


# Observation: as we can see Item_Outlet_Sales and Item_Visibility are having outlier

# In[14]:


q3 = bigmart.Item_Outlet_Sales.quantile(.75) 
q1 = bigmart.Item_Outlet_Sales.quantile(.25) 
iqr = q3 - q1
print(iqr)
upper_extreme = q3 + (1.5*iqr)
lower_extreme = q1 - (1.5*iqr)
print(upper_extreme)
print(lower_extreme)


# In[15]:


bigmart_noOut = bigmart[(bigmart.Item_Outlet_Sales < upper_extreme) & (bigmart.Item_Outlet_Sales > lower_extreme )]


# In[16]:


#caping the outlier
bigmart.loc[(bigmart['Item_Outlet_Sales'] > upper_extreme),bigmart['Item_Outlet_Sales']]= upper_extreme
bigmart.loc[(bigmart['Item_Outlet_Sales'] < lower_extreme),bigmart['Item_Outlet_Sales']]= lower_extreme


# In[17]:


bigmart.Item_Fat_Content.value_counts()


# In[18]:


bigmart.Item_Fat_Content.nunique()


# In[19]:


bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.replace('low fat','LF').replace('LF','Low Fat').replace('reg','Regular')


# In[20]:


bigmart.Item_Fat_Content.value_counts()


# In[21]:


sns.catplot(x = 'Item_Fat_Content', kind = 'count', data = bigmart)


# Oservations: around 64.7% data containing low fat

# In[22]:


bigmart.Item_Visibility.head()


# In[23]:


bigmart.Item_Visibility.median()


# In[24]:


bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median(),inplace = True)


# In[25]:


bigmart.Item_Visibility.describe()


# In[26]:


bigmart.Item_Type.value_counts()


# In[27]:


bigmart.Item_Type.nunique()


# In[ ]:


corr=bigmart.iloc[:,1:].corr()
top_features=corr.index
sns.heatmap(bigmart[top_features].corr(),annot=True)


# In[ ]:


sns.distplot(bigmart['Item_Visibility'])


# In[ ]:


sns.distplot(bigmart['Item_MRP'])


# In[ ]:


sns.distplot(bigmart['Item_Outlet_Sales'])


# In[ ]:


plt.figure(figsize=(6,6))
sns.countplot(x=bigmart.Outlet_Establishment_Year)
plt.show()


# In[ ]:


plt.figure(figsize=(30,6))
sns.countplot(x=bigmart.Item_Type)
plt.show()


# Oservations:
# More than 14%(ie more than 1200 items) are fruits & vegetables and snacks and foods.
# Sale of breakfast and seafood type of items are very less.
