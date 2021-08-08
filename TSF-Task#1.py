#!/usr/bin/env python
# coding: utf-8

# ## DATA SCIENCE AND BUSINESS ANALYTICS INTERN AT THE SPARKS FOUNDATION 

# ## Task 1: Prediction using Supervised ML

# ## Submitted by: Laxman Parab

# ### Problem Statement: What will be predicted score if a student studies for 9.25 hrs/ day?

# ##### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Importing Dataset

# In[2]:


data = pd.read_csv("http://bit.ly/w-data")
data.head()


# #### Understanding Dataset

# In[3]:


data.dtypes


# In[4]:


data.shape


# In[5]:


data.describe()


# #### Data Pre-processing

# In[6]:


# Checking if missing values exist in the dataset.
print(data.isna().sum())


# The sum of all the cells of Hours and Scores is zero which implies that missing values don't exist in our dataset

# In[7]:


# Checking if outliers exist in the datase
sns.boxplot(data = data)


# We don't have any outliers in our dataset.

# In[8]:


# Check if any relationship between the variables of our data.
data.plot(kind='scatter',x='Hours',y='Scores')
plt.title('Hours vs Scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()


# We can see from the above scatter plot there is a positive linear relationship between 'Study Hours' and 'Test Scores'.

# #### Model Fitting

# In[9]:


x=data.drop('Scores',axis=1)   # Target variable 
y=data['Scores']              # Feature varibale

# Splitting dataset into training and test dataset.

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=6,train_size=.80) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[10]:


# Fitting the data.
my_model=LinearRegression()
my_model.fit(X_train,y_train)


# In[11]:


lr_coeff=np.round(my_model.coef_[0],2)
print(lr_coeff)
lr_intercepts=np.round(my_model.intercept_,2)
print(lr_intercepts)
print('Scores=',lr_coeff, '* Hours +',lr_intercepts)


# #### Plotting Regression Line

# In[12]:


plt.plot(x, y, 'o')
plt.plot(x, my_model.coef_*x + my_model.intercept_)


# #### Predictions

# In[13]:


# Predciting scores on test data.
y_pred=my_model.predict(X_test)
y_pred


# In[14]:


# Comparing Actual Scores vs Predicted Scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# ##### Predicted score if a student studies for 9.25 hrs/ day : 

# In[15]:


hours = 9.25
pred = my_model.predict([[hours]])[0]
print('The predicted score if a student studies for', f'{hours} hrs/day is',round(pred,2),'%')


# #### Evaluating the model

# In[16]:


from sklearn.metrics import mean_absolute_error,mean_squared_error 
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

