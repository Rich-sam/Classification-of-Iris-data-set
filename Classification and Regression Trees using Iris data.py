#!/usr/bin/env python
# coding: utf-8

# In this example, we use the Classification and Regression Trees (CART) decision tree algorithm to model the Iris flower dataset.This dataset is provided as an example dataset with the library and is loaded. The classifier is fit on the data and then predictions are made on the training data.

# In[1]:


# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# Running this example produces the following output, showing you the details of the trained model, the skill of the model according to some common metrics and a confusion matrix

# In[3]:


print(dataset)


# In[ ]:




