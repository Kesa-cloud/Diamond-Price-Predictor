#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


git add requirements.txt
git commit -m "Add matplotlib to requirements"
git push origin main
# In[ ]:


Diamond_df = pd.read_csv("C:/Users/samgi/OneDrive/Documents/diamonds.csv")
Diamond_df.head()


# In[ ]:


Diamond_df.describe()


# In[ ]:


Diamond_df.info()


# In[ ]:


row, column=Diamond_df.shape
print("Row of Dataframe:",row)
print("Column of Dataframe", column)


# In[ ]:


Diamond_df.isna().sum()


# In[ ]:


Duplicated_data=Diamond_df.duplicated().sum()
print("Sum Of Duplicated data:", Duplicated_data)


# In[ ]:


categorical_feature=Diamond_df.select_dtypes(include='object').columns
fig, axes=plt.subplots(1, len(categorical_feature), figsize=(14,6))
axes=axes.flatten()
for i, feature in enumerate(categorical_feature):
    sns.countplot(x=Diamond_df[feature],data=Diamond_df, palette='pastel',ax=axes[i])
    axes[i].set_title(f'Feature distribution of {feature}')

plt.tight_layout()
plt.show()


# In[ ]:


numerical_feature=Diamond_df.select_dtypes(include='number').columns
fig, axes=plt.subplots(4, 2, figsize=(16,20))
axes=axes.flatten()
for i, feature in enumerate(numerical_feature):
    sns.boxplot(x=Diamond_df[feature],ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(data=Diamond_df, x='color', hue='clarity')
plt.title('Color by Clarity')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(data=Diamond_df, x='color',hue='cut')
plt.title('Color by Cut')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(data=Diamond_df, x='cut', hue='clarity')
plt.title('Cut by Clarity')
plt.show()


# In[ ]:


comparison=['x','y','z','price']
fig, axes=plt.subplots(2,2, figsize=(15,12))
axes=axes.flatten()
for i, feature in enumerate(comparison):
    sns.boxplot(data=Diamond_df,x='cut',y=feature,ax=axes[i])
    axes[i].set_title(f'Comparison of Diamond dimension {feature} across Cut Grades')
    if feature=='price':
        axes[i].set_title(f'Comparison of Diamond Price Across Cut Grades')

plt.tight_layout()
plt.show()


# In[ ]:


sns.boxplot(x='cut', y='carat', data=Diamond_df)
plt.title('Comparison of Carat Diamond and Cut Grades ')
plt.show()


# In[ ]:


Diamond_clean=Diamond_df.drop(columns='Unnamed: 0')
Diamond_clean.head()


# In[ ]:


def outlier_handling(df,columns):
    df_cleaned=df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            Q1=df_cleaned[col].quantile(0.25)
            Q3=df_cleaned[col].quantile(0.75)
            IQR=Q3-Q1
            lower_bound=Q1- 1.5* IQR
            upper_bound=Q3+1.5 *IQR
            df_cleaned=df_cleaned[(df_cleaned[col] >=lower_bound) & (df_cleaned[col] <=upper_bound)]
    df_cleaned=df_cleaned.reset_index(drop=True)
    return df_cleaned

Diamond_clean=outlier_handling(Diamond_clean,columns=['carat','depth','table','price','x','y','z'])


# In[ ]:


num_clean=Diamond_clean.select_dtypes(include='number').columns
fig,axes=plt.subplots(4,2,figsize=(16,20))
axes=axes.flatten()
for i, feature in enumerate(num_clean):
    sns.boxplot(x=Diamond_clean[feature], ax=axes[i])
    axes[i].set_title(f"Boxplot of {feature}")

for j in range(len(num_clean), len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()


# In[ ]:


clarity_encoding = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}

cut_encoding = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4,
}

color_encoding = {
    "D": 6,
    "E": 5,
    "F": 4,
    "G": 3,
    "H": 2,
    "I": 1,
    "J": 0,
}

Diamond_clean['clarity'] = Diamond_clean['clarity'].map(clarity_encoding)
Diamond_clean['cut'] = Diamond_clean['cut'].map(cut_encoding)
Diamond_clean['color'] = Diamond_clean['color'].map(color_encoding)


Diamond_clean.head()


# In[ ]:


X=Diamond_clean.drop(columns='price', axis=1)
y=Diamond_clean['price']


# In[ ]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
print(f"Data Training Shape : X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Data Testing Shape: X_test {X_test.shape}. y_test {y_test.shape}")


# In[ ]:


LR=LinearRegression().fit(X_train, y_train)
eval_LR=LR.predict(X_test)

mse_LR= round(mean_squared_error(y_test, eval_LR), 3)
mae_LR= round(mean_absolute_error(y_test, eval_LR), 3)
r2_LR= round(r2_score(y_test, eval_LR), 3)


# In[ ]:


data=({
    "MAE":[mae_LR],
    "MSE":[mse_LR],
    "R2 Score":[r2_LR]
})
result=pd.DataFrame(data, index=['Linear Regression'])
result


# In[ ]:


RF=RandomForestRegressor().fit(X_train, y_train)
eval_RF=RF.predict(X_test)

mse_RF= round(mean_squared_error(y_test, eval_RF), 3)
mae_RF= round(mean_absolute_error(y_test, eval_RF), 3)
r2_RF= round(r2_score(y_test, eval_RF), 3)


# In[ ]:


result.loc['Random Forest']=[mae_RF,mse_RF, r2_RF]
result


# In[ ]:


SVM=SVR().fit(X_train, y_train)
eval_SVM= SVM.predict(X_test)

mse_svm= round(mean_squared_error(y_test, eval_SVM),3)
mae_svm= round(mean_absolute_error(y_test, eval_SVM), 3)
r2_svm= round(r2_score(y_test, eval_SVM), 3)


# In[ ]:


result.loc['SVM']= [mae_svm, mse_svm, r2_svm]
result


# In[ ]:


DT=DecisionTreeRegressor().fit(X_train, y_train)
eval_DT= DT.predict(X_test)

mse_dt= round(mean_squared_error(y_test, eval_DT),3)
mae_dt= round(mean_absolute_error(y_test, eval_DT), 3)
r2_dt= round(r2_score(y_test, eval_DT),3)


# In[ ]:


result.loc['Decision Tree']=[mae_dt, mse_dt, r2_dt]
result







