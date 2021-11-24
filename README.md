# Bigmart
- In this repository, I have done Exploratory data analysis and feature engineering for bigmart dataset

## Technologies used:
- Python
- Pandas
- Matplotlib
- Seaborn
## importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### importing dataset

bigmart = pd.read_csv(r"C:\Users\masoo\Downloads\Bigmart.csv")

bigmart.head()

bigmart.shape

### Observations: This shows that the bigmart data is imported successfully and it consists of 8,523 rows with 12 columns.

bigmart[bigmart.Item_Identifier=='FDP10']

bigmart.columns

## Finding some basic information about the features of the data.
bigmart.info()

## Numerical Features: Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales(Target Variable)

## Categorical Features: Item_Identifier, Item_Fat_Content(Ordinal Feature), Item_Type, Outlet_Itemtifier, Outlet_Establishment_Year, Outlet_Size(Ordinal Feature), Outlet__Location_Type(Ordinal Feature), Ootlet_Type(Ordinal Feature)

## Observations: There are 4 float type variables, 1 integer type and 7 object type.
#### We are considering Item_Establishment_Year as a categorical feature because it contains some fixed value but not converting its data type now will consider later.
#### Item_Fat_Content, Outlet_Size, Outelet_Location_Type and Outlet_Type are ordinal features because these values can be arranged in some order.

### description about the data
bigmart.describe()

## checking null values
bigmart.isnull().sum()

## percentage of null values having columns
bigmart.isnull().sum()/bigmart.shape[0]*100

## Observation: we can see that Item_Identifier having 17% of null values and Outlet_Size haveing 28% of null values.

## imputing null values
bigmart.Item_Weight.fillna(bigmart.Item_Weight.mean(),inplace = True)
bigmart.Outlet_Size.fillna('Not Defined',inplace = True)

### droping unwanted columns
bigmart.drop(['Item_Identifier','Outlet_Identifier'], axis = 1, inplace = True)

### checking Outlier
bigmart.boxplot()
plt.boxplot(bigmart.Item_Outlet_Sales )
plt.boxplot(bigmart.Item_Visibility)

### Observation: as we can see Item_Outlet_Sales and Item_Visibility are having outlier

q3 = bigmart.Item_Outlet_Sales.quantile(.75) 
q1 = bigmart.Item_Outlet_Sales.quantile(.25) 
iqr = q3 - q1
print(iqr)
upper_extreme = q3 + (1.5*iqr)
lower_extreme = q1 - (1.5*iqr)
print(upper_extreme)
print(lower_extreme)

bigmart_noOut = bigmart[(bigmart.Item_Outlet_Sales < upper_extreme) & (bigmart.Item_Outlet_Sales > lower_extreme )]

## caping the outlier
bigmart.loc[(bigmart['Item_Outlet_Sales'] > upper_extreme),bigmart['Item_Outlet_Sales']]= upper_extreme
bigmart.loc[(bigmart['Item_Outlet_Sales'] < lower_extreme),bigmart['Item_Outlet_Sales']]= lower_extreme

bigmart.Item_Fat_Content.value_counts()

bigmart.Item_Fat_Content.nunique()

bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.replace('low fat','LF').replace('LF','Low Fat').replace('reg','Regular')

bigmart.Item_Fat_Content.value_counts()

sns.catplot(x = 'Item_Fat_Content', kind = 'count', data = bigmart)

## Oservations: around 64.7% data containing low fat

bigmart.Item_Visibility.head()

bigmart.Item_Visibility.median()

bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median(),inplace = True)

bigmart.Item_Visibility.describe()

bigmart.Item_Type.value_counts()

bigmart.Item_Type.nunique()

corr=bigmart.iloc[:,1:].corr()
top_features=corr.index
sns.heatmap(bigmart[top_features].corr(),annot=True)

sns.distplot(bigmart['Item_Visibility'])

sns.distplot(bigmart['Item_MRP'])

sns.distplot(bigmart['Item_Outlet_Sales'])

plt.figure(figsize=(6,6))
sns.countplot(x=bigmart.Outlet_Establishment_Year)
plt.show()

plt.figure(figsize=(30,6))
sns.countplot(x=bigmart.Item_Type)
plt.show()

## Oservations:
### More than 14%(ie more than 1200 items) are fruits & vegetables and snacks and foods.
### Sale of breakfast and seafood type of items are very less.
