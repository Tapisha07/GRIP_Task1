#!/usr/bin/env python
# coding: utf-8

# ## TAPISHA PUROHIT- TASK:1
# 

# ### Prediction Using supervised Machine Learning

# ### Problem Statement

# Predict the percentage of a student give on study per hour.This a simpl regression model by using two fetaures i.e. Independent and Dependent variable.

# # Importing Libarires

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,r2_score


# ## Step 1:- Loading Datasets

# In[4]:


data = pd.read_csv("score.csv")


# In[5]:


print("Imported Successfully")


# In[7]:


print("{} Samples & {} Features in Datasets". format(data.shape[0], data.shape[1]))


# In[8]:


print(" Sample of Dataset")
data.head()


# In[9]:


print(" Description of Dataset")
data.describe()


# ## Step 2:- Analysing the missing values

# In[11]:


data.isnull().sum()


# We have seen above that there is no missing values in the dataset.

# ## Step 3:- Exploratory Data Analysis

# ### Univariate Analysis

# In[12]:


data.hist()


# We can seen the above graph that he hours & scores are normally distributed.

# ### Bivariate Analysis

# ### Distribution of Score & Hours.

# In[17]:


plt.scatter(x=data['Hours'],y=data["Scores"])
plt.title(" Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel(" Percentage score")


# ## Step 4:- Preparing Dataset

# Here I'm splitting the data into Train & Test

# In[18]:


X = data.iloc[:, :-1].values
Y = data.iloc[:,-1].values


# In[19]:


train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size= 0.2)


# In[20]:


train_x.shape, test_x.shape, train_y.shape, test_y.shape


# ## Step 5:- Model Building- Linear Regression

# In[24]:


model = LinearRegression()


# In[27]:


model.fit(train_x, train_y)
print("Training Completed")


# In[28]:


line = model.intercept_ + model.coef_ * train_x


# In[29]:


plt.scatter(train_x, train_y)
plt.plot(train_x, line, 'r')
plt.title("Actual vs Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")


# ## Step 6:- Evaluating Model

# In[31]:


pred_y = model.predict(test_x)


# In[32]:


pred_y


# In[33]:


test_y


# In[35]:


df=pd.DataFrame({"Actual":test_y, "Predicted": pred_y})


# In[36]:


df


# In[37]:


df.plot()


# In[38]:


print("Test Accuracy:", model.score(test_x,test_y)* 100)


# In[39]:


mean_squared_error(test_y, pred_y)


# In[40]:


mean_absolute_error(test_y, pred_y)


# In[41]:


r2_score(test_y, pred_y)


# ## Prediction of unknown data

# In[43]:


hour1=[[9.25]]
model.predict(hour1)


# So while studying 9.25 hours you can get 90.91 % by prediciton of our  odel.

# ## CONCLUSION

# This model gives us 93 % accuracy.
# 37 % mean squared error.
# 5 % mean absolute error.
