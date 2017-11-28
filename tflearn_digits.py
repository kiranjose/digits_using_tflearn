
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tflearn
import tensorflow as tf
from keras.utils.np_utils import to_categorical


# In[4]:


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("./train.csv")
test  = pd.read_csv("./test.csv")

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
print("Train set has {0[0]} rows and {0[1]} columns".format(train.shape))


# In[5]:


for x in range(1, 2):
    A=np.array(train[x-1:x])
    c=np.resize(A,(1,1))
    print('index shape             ->',c.shape)
    print('index                   ->',c.ravel())
    rowWithIndex = A.ravel()#scalar 785    
    print('row with index shape    ->',rowWithIndex.shape)
    rowWithOutIndex = rowWithIndex[1:785:1]#scalar 784 skip the index for image
    print('row without index shape ->',rowWithOutIndex.shape)
    Image1=np.resize(rowWithOutIndex,(28,28))
    print('Image shape             ->',Image1.shape)
    plt.imshow(Image1, interpolation='nearest')
    plt.show()


# In[6]:


# Split data into training set and validation set
y_train = train.ix[:,0].values #all input labels, first cloumn(index 0) of each row in the train csv file
trainX = train.ix[:,1:].values #remaining 784 values after(from index 1 till end) the first colum. 

print(y_train.shape)
print(trainX.shape)
#one hot encoded form of labels
y_train_one_hot = to_categorical(y_train)
print(y_train_one_hot)


# In[7]:


#DNN - input layer of 784 inputs, 4 hidden layers and a softmax layer at output
def build_model():
    tf.reset_default_graph()
    net = tflearn.input_data([None, 784])
    net = tflearn.fully_connected(net, 128, activation='ReLU')
    net = tflearn.fully_connected(net, 64, activation='ReLU')
    net = tflearn.fully_connected(net, 32, activation='ReLU')
    net = tflearn.fully_connected(net, 10, activation='softmax') 
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model
model = build_model()


# In[8]:


#training
model.fit(trainX, y_train_one_hot, validation_set=0.1, show_metric=True, batch_size=300, n_epoch=10)


# In[ ]:


#inference
testX = test.ix[:,0:].values
def prediction(predictions):
    return np.argmax(predictions,1)
predictions = prediction(model.predict(testX))


# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)
print(submissions)

