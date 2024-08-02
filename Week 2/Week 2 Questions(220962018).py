#!/usr/bin/env python
# coding: utf-8

# ### Question 1

# In[22]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x=[0,20,60,100]
y=[0,40,120,200]
ax.plot(x,y)
ax.set_xlabel('x') 
ax.set_ylabel('y') 
ax.set_title('title') 


# ### Question 2

# In[30]:


fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])
x=[0,20,60,100]
y=[0,40,120,200]
ax1.plot(x,y)
ax2.plot(x,y)
ax1.set_xlabel('x1') 
ax1.set_ylabel('y1') 
ax1.set_title('title1')
ax2.set_xlabel('x2') 
ax2.set_ylabel('y2') 
ax2.set_title('title2')


# ### Question 3 

# In[31]:


import pandas as pd
data=pd.read_csv('company_sales_data.csv')
data.head()


# In[32]:


x=data["month_number"]
y=data["total_profbit"]
plt.xlabel("Month Number")
plt.ylabel("Total Profit")
plt.plot(x,y)
plt.show()


# ### Question 4

# In[33]:


x=data["month_number"]
y=data["total_units"]
plt.xlabel("Month Number")
plt.ylabel("Sold Units Number")
plt.plot(x,y,linestyle='dotted',color='red',marker='o',linewidth=3)
plt.legend(["Total Units"],loc="lower right")
plt.show()


# ### Addtional Question 1

# In[46]:


x=data["month_number"]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

for product in ['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']:
    ax.plot(x, data[product], label=product)
    
ax.set_xlabel('Month Number')
ax.set_ylabel('Units Sold')
ax.set_title('Units Sold per Month for Each Product')

ax.legend()
plt.show()


# ### Addtional Question 2

# In[44]:


total_units_sold = data[['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']].sum()
plt.pie(total_units_sold.values, labels=total_units_sold.index, autopct='%1.1f%%')
plt.title('Total Sales Data')
plt.show()

