#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow


# In[2]:


from tensorflow.keras.optimizers import SGD


# In[3]:


from tensorflow.keras.models import Sequential


# In[4]:


from tensorflow.keras.applications import VGG16


# In[5]:


from tensorflow.keras.layers import Input, Dense,Dropout,GlobalAveragePooling2D


# In[6]:


np.random.seed(42)
tensorflow.random.set_seed(42)


# In[7]:


train_dataset = h5py.File('/cxldata/datasets/project/cat-non-cat/train_catvnoncat.h5', "r")

test_dataset = h5py.File('/cxldata/datasets/project/cat-non-cat/test_catvnoncat.h5', "r")

print("File format of train_dataset:",train_dataset)
print("File format of test_dataset:",test_dataset)


# In[8]:


train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels


# In[9]:


classes = np.array(test_dataset["list_classes"][:])

print("Classes are: ",classes)
print("Groundtruth stored as: ",train_set_y_orig[2])
print(classes[train_set_y_orig[2]].decode('utf-8'))

plt.imshow(train_set_x_orig[2])


# In[12]:


validation_x = test_set_x_orig[:25]
validation_y = test_set_y_orig[:25]

test_set_x = test_set_x_orig[25:]
test_set_y = test_set_y_orig[25:]


# In[13]:


print("train_set_x shape: ", train_set_x_orig.shape)
print("train_set_y shape: ", train_set_y_orig.shape)


# In[15]:


print("Validation data size: ", validation_x.shape)
print("Validation data size: ", validation_y.shape)


# In[16]:


print("test_set_x shape: ", test_set_x.shape)
print("test_set_y shape: ", test_set_y.shape)


# In[30]:


vgg_base = VGG16(weights='imagenet', include_top=False)

vgg_base.trainable=False


# In[35]:


inp = Input(shape=(64, 64, 3), name='image_input')


# In[36]:


#initiate a model
vgg_model = Sequential()


# In[37]:


#Add the VGG base model
vgg_model.add(vgg_base)


# In[38]:


vgg_model.add(GlobalAveragePooling2D())

vgg_model.add(Dense(1024,activation='relu'))
vgg_model.add(Dropout(0.6))

vgg_model.add(Dense(512,activation='relu'))
vgg_model.add(Dropout(0.5))

vgg_model.add(Dense(1024,activation='relu'))
vgg_model.add(Dropout(0.4))

vgg_model.add(Dense(1024,activation='relu'))
vgg_model.add(Dropout(0.3))

vgg_model.add(Dense(1, activation='sigmoid'))


# In[44]:


sgd = SGD(learning_rate=0.025)


# In[47]:


vgg_model.compile(loss='binary_crossentropy', optimizer=sgd,    metrics=['accuracy'])


# In[52]:


vgg_model.fit(train_set_x_orig, train_set_y_orig, epochs=10, verbose=1, validation_data=(validation_x, validation_y))


# In[55]:


vgg_model_loss, vgg_model_acc = vgg_model.evaluate(test_set_x_orig,test_set_y_orig)


# In[56]:


print('Test accuracy using VGG16 model as the base:', vgg_model_acc)


# In[60]:


vgg_base.summary()


# In[61]:


from tensorflow.keras.utils import plot_model


# In[62]:


plot_model(vgg_base,show_shapes=True, show_layer_names=True)


# In[63]:


vgg_model.summary()


# In[64]:


plot_model(vgg_model,show_shapes= True, show_layer_names=True)


# In[ ]:




