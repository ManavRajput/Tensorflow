#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])


# In[14]:


def mae(y_true,y_predicted):
    total_error = 0
    for yt, yp in zip(y_true,y_predicted):
        total_error += abs(yt - yp)
    print("Total Error = ",total_error)
    mae = total_error/len(y_true)
    print("Mean absolute Error = ",mae)
    return mae


# In[15]:


mae(y_true,y_predicted)


# In[16]:


np.mean(np.abs(y_true - y_predicted))


# In[19]:


np.log([0.1]) #log of zero is not defined so we reduce the value using elsilon 


# In[ ]:





# In[51]:


epsilon = 1e-15 #we will either sub or add the epsilon to the original value to 
#reduce it to != 0


# In[52]:


y_predicted_new = [max(i,epsilon) for i in y_predicted]


# In[53]:


y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
y_predicted_new


# In[54]:


y_predicted_new = np.array(y_predicted_new)
np.log(y_predicted_new)


# In[55]:


-np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-(y_predicted_new)))


# In[56]:


def log_loss(y_true, y_predicted):   # log Loss or binary_crossentropy error
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    loss = -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-(y_predicted_new)))
    
    return loss


# In[57]:


log_loss(y_true, y_predicted)


# In[61]:


np.mean(np.square(y_true - y_predicted))


# In[62]:


def mse(y_true,y_predicted):
    mean = 0
    for yt, yp in zip(y_true,y_predicted):
        mean += (yt - yp)**2/len(y_true)
    return mean


# In[64]:


mse(y_true,y_predicted)


# In[ ]:




