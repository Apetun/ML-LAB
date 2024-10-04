#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[32]:


import math
import numpy as np
import pandas as pd
from collections import Counter


# ### Creating Dataset

# In[34]:


data = {
    "Outlook":["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temp":[85,80,83,70,68,65,64,72,69,75,75,72,81,71],
    "Humidity":[85,90,78,96,80,70,65,95,70,80,70,90,75,80],
    "Wind":["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Decision":["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)
print(df)


# ### Question 1

# In[35]:


# Calculate entropy
def entropy(data):
    total_count = len(data)
    label_counts = Counter(row[-1] for row in data)
    return -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())

# Calculate information gain
def information_gain(data, feature_index):
    total_entropy = entropy(data)
    feature_values = set(row[feature_index] for row in data)
    weighted_entropy = 0

    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# C4.5 Decision Tree Implementation
def c45(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not features:
        return Counter(labels).most_common(1)[0][0]

    gains = [information_gain(data, feature) for feature in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in set(row[best_feature] for row in data):
        subset = [row for row in data if row[best_feature] == value]
        tree[best_feature][value] = c45(subset, remaining_features)

    return tree


# ### Question 2

# In[36]:


# Calculate Gini impurity
def gini_impurity(data):
    total_count = len(data)
    if total_count == 0:
        return 0
    label_counts = Counter(row[-1] for row in data)
    return 1 - sum((count / total_count) ** 2 for count in label_counts.values())

# Calculate Gini Gain
def gini_gain(data, feature_index):
    total_gini = gini_impurity(data)
    feature_values = set(row[feature_index] for row in data)
    weighted_gini = 0

    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        weighted_gini += (len(subset) / len(data)) * gini_impurity(subset)

    return total_gini - weighted_gini

# CART Decision Tree Implementation
def cart(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not features:
        return Counter(labels).most_common(1)[0][0]

    gains = [gini_gain(data, feature) for feature in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in set(row[best_feature] for row in data):
        subset = [row for row in data if row[best_feature] == value]
        tree[best_feature][value] = cart(subset, remaining_features)

    return tree


# ### Results

# In[37]:


# Main execution
data_list = df.values.tolist()  # Convert DataFrame to list of lists
features = list(range(len(df.columns) - 1))  # Use indices for features (0, 1, 2, 3)

# Build C4.5 and CART trees
c45_tree = c45(data_list, features)
cart_tree = cart(data_list, features)


# Display the trees
print("C4.5 Decision Tree:")
print(c45_tree)

print("\nCART Decision Tree:")
print(cart_tree)


# ### Question 3

# In[38]:


import pandas as pd

# Create the dataset
data = {
    "Income": ["Low", "Low", "Medium", "Medium", "High", "High"],
    "Credit": ["Good", "Bad", "Good", "Bad", "Good", "Bad"],
    "Loan Approved": ["Yes", "No", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
print(df)


# ### Function for C4.5

# In[39]:


import math
from collections import Counter

def entropy(data):
    total_count = len(data)
    label_counts = Counter(row[-1] for row in data)
    return -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())

def information_gain(data, feature_index):
    total_entropy = entropy(data)
    feature_values = set(row[feature_index] for row in data)
    weighted_entropy = 0

    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

def c45(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not features:
        return Counter(labels).most_common(1)[0][0]

    gains = [information_gain(data, feature) for feature in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in set(row[best_feature] for row in data):
        subset = [row for row in data if row[best_feature] == value]
        tree[best_feature][value] = c45(subset, remaining_features)

    return tree


# ### Function for CART

# In[40]:


def gini_impurity(data):
    total_count = len(data)
    if total_count == 0:
        return 0
    label_counts = Counter(row[-1] for row in data)
    return 1 - sum((count / total_count) ** 2 for count in label_counts.values())

def gini_gain(data, feature_index):
    total_gini = gini_impurity(data)
    feature_values = set(row[feature_index] for row in data)
    weighted_gini = 0

    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        weighted_gini += (len(subset) / len(data)) * gini_impurity(subset)

    return total_gini - weighted_gini

def cart(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not features:
        return Counter(labels).most_common(1)[0][0]

    gains = [gini_gain(data, feature) for feature in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in set(row[best_feature] for row in data):
        subset = [row for row in data if row[best_feature] == value]
        tree[best_feature][value] = cart(subset, remaining_features)

    return tree


# ### Predictions

# In[41]:


# Prepare the data
data_list = df.values.tolist()
features = list(range(len(df.columns) - 1))  # Indices of features

# Build C4.5 and CART trees
c45_tree = c45(data_list, features)
cart_tree = cart(data_list, features)

# Function to print the decision tree
def print_tree(tree, level=0):
    if isinstance(tree, dict):
        for feature, branches in tree.items():
            for branch, subtree in branches.items():
                print("    " * level + f"{feature} = {branch}:")
                print_tree(subtree, level + 1)
    else:
        print("    " * level + f"Predict: {tree}")

# Print C4.5 Decision Tree
print("\nC4.5 Decision Tree:")
print_tree(c45_tree)

# Print CART Decision Tree
print("\nCART Decision Tree:")
print_tree(cart_tree)


# ### Using SckitLearn

# In[43]:


from sklearn.tree import DecisionTreeClassifier, export_text

# Prepare features and target variable
X = df[["Income", "Credit"]]
y = df["Loan Approved"]

# Convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)

# Create and fit the C4.5 decision tree
clf_c45 = DecisionTreeClassifier(criterion='entropy')
clf_c45.fit(X, y)

# Create and fit the CART decision tree
clf_cart = DecisionTreeClassifier(criterion='gini')
clf_cart.fit(X, y)

# Print the tree structure
print("\nC4.5 Decision Tree (Scikit-Learn):")
print(export_text(clf_c45, feature_names=list(X.columns)))

print("\nCART Decision Tree (Scikit-Learn):")
print(export_text(clf_cart, feature_names=list(X.columns)))

