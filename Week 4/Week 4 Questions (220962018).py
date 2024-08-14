#!/usr/bin/env python
# coding: utf-8

# ### Question 1

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Create a CSV file with sample data

# In[107]:


data = {
    'Temp': [50, 50, 50, 70, 70, 70, 80, 80, 80, 90, 90, 90, 100, 100, 100],
    'Yield': [3.3, 2.8, 2.9, 2.3, 2.6, 2.1, 2.5, 2.9, 2.4, 3.0, 3.1, 2.8, 3.3, 3.5, 3.0]
}

df = pd.DataFrame(data)

csv_file = 'yield_data.csv'
df.to_csv(csv_file, index=False)


# In[108]:


X=df['Temp'].to_numpy()
Y=df['Yield'].to_numpy()


# ### Pedhazur Linear Method

# In[109]:


b1_pedahzur=np.sum((X-X.mean())*(Y-Y.mean()))/np.sum((X-X.mean())**2)
b0_pedahzur=Y.mean()-(b1_pedahzur*X.mean())

y_pedahzur = b0_pedahzur + b1_pedahzur * X
squared_errors = (Y - y_pedahzur) ** 2
mse_pedahzur = np.mean(squared_errors)
rmse_pedahzur = np.sqrt(np.mean(squared_errors))

print(f"Equation : {b1_pedahzur}x + {b0_pedahzur}")
print(f"Predicted y : {y_pedahzur}")
print(f"MSE : {mse_pedahzur}")
print(f"RMSE : {rmse_pedahzur}")


# ### Matrix Linear Method

# In[110]:


mat1=np.array([[len(X),np.sum(X)],
               [np.sum(X),np.sum(X**2)]])
mat2=np.array([[np.sum(Y),
                np.sum(X*Y)]])
coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
b0_mat,b1_mat=coeffs[0,0],coeffs[1,0]

y_mat1 = b0_mat + b1_mat * X
squared_errors = (Y - y_mat1) ** 2
mse_mat1 = np.mean(squared_errors)
rmse_mat1 = np.sqrt(np.mean(squared_errors))

print(f"Equation : {b1_mat}x + {b0_mat}")
print(f"Predicted y : {y_mat1}")
print(f"MSE : {mse_mat1}")
print(f"RMSE : {rmse_mat1}")


# ### Matrix Polynomial Method

# In[111]:


mat1=np.array([[len(X),np.sum(X),np.sum(X**2)],
               [np.sum(X),np.sum(X**2),np.sum(X**3)],
               [np.sum(X**2),np.sum(X**3),np.sum(X**4)]])
mat2=np.array([[np.sum(Y),
                np.sum(X*Y),
                np.sum((X**2)*Y)]])
coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
b0_mat,b1_mat,b2_mat=coeffs[0,0],coeffs[1,0],coeffs[2,0]

y_mat2 = b0_mat + b1_mat * X + b2_mat * (X**2)
squared_errors = (Y - y_mat2) ** 2
mse_mat2 = np.mean(squared_errors)
rmse_mat2 = np.sqrt(np.mean(squared_errors))

print(f"Equation : {b2_mat}x^2 + {b1_mat}x + {b0_mat}")
print(f"Predicted y : {y_mat2}")
print(f"MSE : {mse_mat2}")
print(f"RMSE : {rmse_mat2}")


# ### Plotting Results

# In[112]:


plt.figure(figsize=(12, 8))
plt.scatter(X, Y, color='black', label='Original Data')
plt.plot(X, y_mat1, color='green', label=f'Linear Model (Matrix): RMSE={rmse_mat1:.2f} , MSE={mse_mat1:.2f}')
X_range = np.linspace(min(X), max(X), 100)
y_range = b0_mat + b1_mat * X_range + b2_mat * (X_range ** 2)
plt.plot(X_range, y_range, color='red', label=f'Quadratic Model: RMSE={rmse_mat2:.2f}, MSE={mse_mat2:.2f}')

plt.xlabel('Temp')
plt.ylabel('Yield')
plt.title('Results')
plt.legend()
plt.show()


# ### Question 2

# ### Create a CSV file with sample data

# In[113]:


data = {
    'Infarc': [0.119, 0.19, 0.395, 0.469, 0.13, 0.311, 0.418, 0.48, 0.687, 0.847,
               0.062, 0.122, 0.033, 0.102, 0.206, 0.249, 0.22, 0.299, 0.35, 0.35,
               0.588, 0.379, 0.149, 0.316, 0.39, 0.429, 0.477, 0.439, 0.446, 0.538,
               0.625, 0.974],
    'Area': [0.34, 0.64, 0.76, 0.83, 0.73, 0.82, 0.95, 1.06, 1.2, 1.47,
             0.44, 0.77, 0.9, 1.07, 1.01, 1.03, 1.16, 1.21, 1.2, 1.22,
             0.99, 0.77, 1.05, 1.06, 1.02, 0.99, 0.97, 1.12, 1.23, 1.19,
             1.22, 1.4],
    'Group': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2],
    'X2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
    'X3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1]
}

df = pd.DataFrame(data)
csv_filename = 'rabbit.csv'
df.to_csv(csv_filename, index=False)


# In[114]:


X1=df['Area'].to_numpy()
X2=df['X2'].to_numpy()
X3=df['X3'].to_numpy()
Y=df['Infarc'].to_numpy()


# ### Matrix Method Multi Variable Linear Regression

# In[115]:


mat1=np.array([[len(X1),np.sum(X1),np.sum(X2),np.sum(X3)],
               [np.sum(X1),np.sum(X1**2),np.sum(X1*X2),np.sum(X1*X3)],
               [np.sum(X2),np.sum(X1*X2),np.sum(X2**2),np.sum(X2*X3)],
               [np.sum(X3),np.sum(X1*X3),np.sum(X2*X3),np.sum(X3**2)]
              ])
mat2=np.array([[np.sum(Y),
                 np.sum(X1*Y),
                np.sum(X2*Y),
                np.sum(X3*Y)
               ]])
coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
b0,b1,b2,b3=coeffs[0,0],coeffs[1,0],coeffs[2,0],coeffs[3,0]

y = b0 + b1 * X1 + b2 * X2 + b3 * X3
squared_errors = (Y - y) ** 2
mse_mat = np.mean(squared_errors)
rmse_mat = np.sqrt(np.mean(squared_errors))

print(f"Equation : {b3}x3 + {b2}x2 + {b1}x1 + {b0}")
print(f"Predicted y : {y}")
print(f"MSE : {mse_mat}")
print(f"RMSE : {rmse_mat}")


# In[116]:


fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, Y, c='r', marker='o', label='Actual data')
x1_range = np.linspace(min(X1), max(X1), 10)
x2_range = np.linspace(min(X2), max(X2), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x3_grid = np.linspace(min(X3), max(X3), 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
y_grid = b0 + b1 * X1_grid + b2 * X2_grid + b3 * np.mean(X3)
ax.plot_surface(X1_grid, X2_grid, y_grid, color='b', alpha=0.5)

ax.set_xlabel('Area (X1)')
ax.set_ylabel('X2')
ax.set_zlabel('Infarc')
ax.set_title('3D Scatter Plot and Regression Plane')

plt.legend()
plt.show()

