#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


(X_train , y_train) , (X_test , y_test) = keras.datasets.mnist.load_data() 


# In[4]:


X_train = X_train/255
X_test = X_test/255


# In[5]:


len(X_test)


# In[6]:


X_train[0].shape


# In[7]:


plt.matshow(X_train[2])


# In[8]:


y_train[2]


# In[9]:


X_test.shape


# In[10]:


X_train_flatened = X_train.reshape(len(X_train),28*28)
X_test_flatened = X_test.reshape(len(X_test),28*28)


# In[11]:


X_train_flatened.shape


# In[12]:


X_test_flatened.shape


# In[13]:


model = keras.Sequential([
    keras.layers.Dense(10,input_shape = (784,),activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam' ,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(X_train_flatened, y_train, epochs =5)


# In[14]:


model.evaluate(X_test_flatened, y_test)


# In[15]:


plt.matshow(X_test[5])


# In[17]:


y_predicted = model.predict(X_test_flatened)
y_predicted[5]


# In[ ]:


np.argmax(y_predicted[5])


# In[ ]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[ ]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[ ]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(500,input_shape = (784,),activation = 'relu'),
    keras.layers.Dense(250,activation = 'relu'),
    keras.layers.Dense(10,activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam' ,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(X_train_flatened, y_train, epochs =5)


# In[ ]:


model.evaluate(X_test_flatened, y_test)


# In[ ]:


plt.matshow(X_test[5])


# In[ ]:


y_predicted = model.predict(X_test_flatened)
y_predicted[5]


# In[ ]:


np.argmax(y_predicted[5])


# In[ ]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[ ]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[ ]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(500,input_shape = (784,),activation = 'relu'),
    keras.layers.Dense(250,activation = 'tanh'),
    keras.layers.Dense(10,activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam' ,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(X_train, y_train, epochs = 5)


# In[ ]:





# In[ ]:





# In[ ]:




