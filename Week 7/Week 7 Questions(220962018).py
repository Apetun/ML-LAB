#!/usr/bin/env python
# coding: utf-8

# ### Question 1
# 
# ####  Implement in python program of the following problems using Bayes Theorem.

# a) Of the students in the college, 60% of the students reside in the hostel and 40% of the students are day scholars. Previous year results report that 30% of all students who stay in the hostel scored A Grade and 20% of day scholars scored A grade. At the end of the year, one student is chosen at random and found that he/she has an A grade. What is the probability that the student is a hosteler?

# In[52]:


p_hostel = 0.6
p_ds = 0.4
p_A_hostel = 0.3
p_A_ds = 0.2
p_A = (p_A_hostel*p_hostel)+(p_A_ds*p_ds)
p_hostel_A = (p_A_hostel*p_hostel)/p_A
print(f"Probability that the student is a hosteler given A grade : {p_hostel_A:0.3f}")


# b) Suppose you're testing for a rare disease, and you have the following information:
# - The disease has a prevalence of 0.01 (1% of the population has the disease). 
# 
# The test is not perfect:
#  - The test correctly identifies the disease (true positive) 99% of the time (sensitivity).
#  - The test incorrectly indicates the disease (false positive) 2% of the time (1 - specificity).
# Calculate the probability of having the disease given a positive test result using Bayes' theorem.

# ### Question 2

# In[53]:


p_dis = 0.01
p_nd = 0.99
sens = 0.99
spec = 0.02
p_pos_dis = sens
p_pos = ((p_pos_dis)*p_dis) + ((spec)*p_nd)
p_dis_pos = (p_pos_dis*p_dis)/p_pos
print(f"probability of having the disease given a positive test result : {p_dis_pos:0.3f}")


# In[54]:


import numpy as np 
import pandas as pd 


# ### Read data

# In[55]:


df = pd.read_csv("data.csv")
print(df)


# ### For P(E/buy computer = yes)
# ### Calculating probabilities

# In[56]:


count = 0
tot = 14
x = df['buys_com'].values
for i in x:
    if i=='yes':
        count = count + 1
count_yes = count
p_yes = count/tot

y = df['age'].values
count = 0
for i,j in zip(y,x):
    if i=='<=30' and j=='yes':
        count = count + 1
p_c1_yes = count/count_yes

z = df['income'].values
count = 0
for i,j in zip(z,x):
    if i=='medium' and j=='yes':
        count = count+1
p_c2_yes = count/count_yes

a = df['student'].values
count = 0
for i,j in zip(a,x):
    if i=='yes' and j=='yes':
        count = count+1
p_c3_yes = count/count_yes

b = df['credit_rating'].values
count = 0
for i,j in zip(b,x):
    if i=='fair' and j=='yes':
        count = count+1
p_c4_yes = count/count_yes


# ### Calculating P(E1,E2...Ei/buy computer = yes)

# In[57]:


p_E_yes = (p_c1_yes*p_c2_yes*p_c3_yes*p_c4_yes)
p_yes_E = p_E_yes*p_yes
print(f"Probability that the player buys computer  : {p_yes_E}")


# ### For P(E/buy computer = no)
# ### Calculating probabilities

# In[58]:


count = 0
for i in x:
    if i=='no':
        count = count + 1
count_no = count
p_no = count/tot

y = df['age'].values
count = 0
for i,j in zip(y,x):
    if i=='<=30' and j=='no':
        count = count + 1
p_c1_no = count/count_no

z = df['income'].values
count = 0
for i,j in zip(z,x):
    if i=='medium' and j=='no':
        count = count+1
p_c2_no = count/count_no

a = df['student'].values
count = 0
for i,j in zip(a,x):
    if i=='yes' and j=='no':
        count = count+1
p_c3_no = count/count_no

b = df['credit_rating'].values
count = 0
for i,j in zip(b,x):
    if i=='fair' and j=='no':
        count = count+1
p_c4_no = count/count_no


# ### Calculating P(E1,E2...Ei/buy computer = no)

# In[59]:


p_no_E = 0
p_E_no = (p_c1_no*p_c2_no*p_c3_no*p_c4_no)
p_no_E = p_E_no*p_no
print(f"Probability that the player didn't buy computer : {p_no_E}")


# ### Classifying

# In[60]:


if(p_yes_E >= p_no_E):
    print("Classification : Buys computer ")
else:
    print("Classification : Didn't buy computer ")


# ### Question 3

# ### Read Data

# In[61]:


df = pd.read_csv("data2.csv")
print(df)


# ### Getting word vocab

# In[62]:


x = df['Text'].values
y = df['Tag'].values

voc = set()
s_word_count = 0
ns_word_count = 0
for i,j in zip(x,y):
    if j=='Sports':
        w = i.strip('\"').lower().split(" ")
        s_word_count += len(w)
    else:
        w = i.strip('\"').lower().split(" ")
        ns_word_count += len(w)
    for word in w:
            voc.add(word)


# ### Defining functions for finding probabilities

# In[63]:


tot = len(voc)
def find_word_s(word):
    count = 0
    for i,j in zip(x,y):
        w = i.strip('\"').lower().split(" ")
        if j=='Sports' and word in w:
            count = count + 1
    if count == 0:
        return 1/(s_word_count+tot)
    else:
        return (count+1)/(s_word_count+tot) 

def find_word_ns(word):
    count = 0
    for i,j in zip(x,y):
        w = i.strip('\"').lower().split(" ")
        if j=='Not sports' and word in w:
            count = count + 1
    if count == 0:
        return 1/(ns_word_count+tot)
    else:
        return (count+1)/(ns_word_count+tot) 
    


# ### Finding probability for 'a very close game'

# In[64]:


str = 'a very close game'
sent = str.split()

p_word_s = [find_word_s(i) for i in sent]
print(p_word_s)
p_word_ns = [find_word_ns(i) for i in sent]
print(p_word_ns)


# In[65]:


s_count = 0
ns_count = 0
for i in y:
    if i=='Sports':
        s_count = s_count + 1
    else:
        ns_count = ns_count + 1
p_s = s_count/len(y)
p_ns = ns_count/len(y)


# In[66]:


p_text_s = 1
for i in p_word_s:
    p_text_s = p_text_s * i
    
p_s_text = p_text_s*p_s
print(p_s_text)

p_text_ns = 1
for i in p_word_ns:
    p_text_ns = p_text_ns * i
    
p_ns_text = p_text_ns*p_ns
print(p_ns_text)


# ### Classifying using probabilities

# In[67]:


if(p_s_text >= p_ns_text):
    print("Classification : Sports")
else:
    print("Classification : Not Sports")

