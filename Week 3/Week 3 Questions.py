#!/usr/bin/env python
# coding: utf-8

# In[317]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Question 1

# ### Hepatitis Dataset

# In[318]:


data=pd.read_csv('hepatitis_csv.csv')
print(data.columns)
print(data.shape)
data.head()


# In[319]:


mapping = {
    'male': 1,
    'female': 0,
    'live': 1,
    'die': 0
}

for column in data.columns:
    unique_values = data[column].unique()
    if set(unique_values).issubset(set(mapping.keys())):
        data[column] = data[column].map(mapping)
data.head()


# In[320]:


data = data.fillna(data.median())
data.info()


# In[321]:


X = data.drop('class', axis=1).to_numpy() 
y = data['class'].to_numpy()  


# In[322]:


indices = np.arange(X.shape[0])
test_size = 0.2
split_index = int(X.shape[0] * (1 - test_size))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

data.to_csv('hepatitis_clean.csv', index=False)


# ### Question 2

# #### a. Construct a CSV file with the following attributes: Study time in hours of ML lab course (x) , Score out of 10 (y) , The dataset should contain 10 rows.

# In[323]:


data = {
    'StudyTime': [1, 2, 3, 4, 5, 7, 8, 9, 11, 12], 
    'Score': [2, 3, 5, 6, 7, 8, 8, 9, 10, 11]       
}

df = pd.DataFrame(data)
df.to_csv('study_score.csv', index=False)


# #### b. Create a regression model and display the following:Coefficients: B0 (intercept) and B1 (slope) , RMSE (Root Mean Square Error) , Predicted responses

# In[324]:


X=df['StudyTime'].to_numpy()
Y=df['Score'].to_numpy()


# #### Pedahzur Formula

# In[325]:


b1_pedahzur=np.sum((X-X.mean())*(Y-Y.mean()))/np.sum((X-X.mean())**2)
b0_pedahzur=Y.mean()-(b1_pedahzur*X.mean())

y_pedahzur = b0_pedahzur + b1_pedahzur * X
squared_errors = (Y - y_pedahzur) ** 2
rmse_pedahzur = np.sqrt(np.mean(squared_errors))

print(f"{b1_pedahzur}x + {b0_pedahzur}")
print(y_pedahzur)
print(rmse_pedahzur)


# #### Calculus Method

# In[326]:


mat1=np.array([[len(X),np.sum(X)],[np.sum(X),np.sum(X**2)]])
mat2=np.array([[np.sum(Y),np.sum(X*Y)]])
coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
b0_mat,b1_mat=coeffs[0,0],coeffs[1,0]

y_mat = b0_mat + b1_mat * X
squared_errors = (Y - y_mat) ** 2
rmse_mat = np.sqrt(np.mean(squared_errors))

print(f"{b1_mat}x + {b0_mat}")
print(y_mat)
print(rmse_mat)


# #### Predicted Output using sklearn

# In[327]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['StudyTime']].values 
y = df['Score'].values 
model = LinearRegression()
model.fit(X, y)
B1 = model.coef_[0]
B0 = model.intercept_

y_sklearn = model.predict(X)
rmse_sklearn = np.sqrt(mean_squared_error(y, y_sklearn))


print(f"scikit-learn - B0 (Intercept): {B0}")
print(f"scikit-learn - B1 (Slope): {B1}")
print(y_sklearn)
print(rmse_sklearn)


# In[328]:


plt.figure(figsize=(12, 8))


plt.scatter(X, Y, color='red', label='Data Points')
plt.plot(X, y_pedahzur, color='blue', label=f'Pedahzur: {b1_pedahzur}x + {b0_pedahzur}')

plt.xlabel('StudyTime')
plt.ylabel('Score')
plt.title('Scatter Plot with Regression Lines')
plt.legend()
plt.show()


# ### Additional Question

# In[329]:


df = pd.read_csv('hepatitis_clean.csv')
X = df['bilirubin'].to_numpy()  
Y = df['age'].to_numpy()  


# In[330]:


b1_pedahzur=np.sum((X-X.mean())*(Y-Y.mean()))/np.sum((X-X.mean())**2)
b0_pedahzur=Y.mean()-(b1_pedahzur*X.mean())

y_pedahzur = b0_pedahzur + b1_pedahzur * X
squared_errors = (Y - y_pedahzur) ** 2
rmse_pedahzur = np.sqrt(np.mean(squared_errors))

print(f"{b1_pedahzur}x + {b0_pedahzur}")
print(y_pedahzur)
print(rmse_pedahzur)


# In[331]:


mat1=np.array([[len(X),np.sum(X)],[np.sum(X),np.sum(X**2)]])
mat2=np.array([[np.sum(Y),np.sum(X*Y)]])
coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
b0_mat,b1_mat=coeffs[0,0],coeffs[1,0]

y_mat = b0_mat + b1_mat * X
squared_errors = (Y - y_mat) ** 2
rmse_mat = np.sqrt(np.mean(squared_errors))

print(f"{b1_mat}x + {b0_mat}")
print(y_mat)
print(rmse_mat)


# In[332]:


plt.figure(figsize=(12, 8))


plt.scatter(X, Y, color='red', label='Data Points')
plt.plot(X, y_pedahzur, color='blue', label=f'Pedahzur: {b1_pedahzur}x + {b0_pedahzur}')

plt.xlabel('Age')
plt.ylabel('Bilirubin')
plt.title('Scatter Plot with Regression Lines')
plt.legend()
plt.show()

