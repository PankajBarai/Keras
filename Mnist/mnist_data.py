import numpy as np
import matplotlib.pyplot as plt
import tensorflow  as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.layers.core import Dense
from keras import Sequential
from keras.layers import Dropout
from keras.optimizers import Adam

import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-b','--batch_size',type = int,required= True,
                help='#batch size')
ap.add_argument('-e','--epochs', type = int,required=True, 
                help='#epochs')
args = vars(ap.parse_args())
#use --batch_size 64 --epoch 30 for good results or otherwise you can perform trail and error method.


#loading mnist datasets
print('[INFO] loading the MNIST Dataset')
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(x_train.shape)
# print(x_test.shape)

#Data Preprocessing
print('[INFO] data preprocessing ')
RESHAPED = 784 # 28x28 = 784 neurons
x_train = x_train.reshape(60000, RESHAPED) 
x_test = x_test.reshape(10000, RESHAPED) 
x_train = x_train.astype('float')/255.0
x_test = x_test.astype('float')/255.0
print(x_train.shape[0],'train samples')
print(x_test.shape[0], 'test samples')


#one hot represntation of the labels 
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Now the model will take as input arrays of shape (*, 784)# and output arrays of shape (*, 10)
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (RESHAPED,), name = 'Dense_layer_1'))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu', name = 'Dense_layer_2'))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu', name = 'Dense_layer_3'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax', name = 'Output_layer'))


#train the model using compile functions anf fit functions
print('[INFO] training the network...')
#compile
model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics= ['accuracy'])
#fit
History = model.fit(x_train , y_train , batch_size = args['batch_size'], epochs = args['epochs'],validation_split= 0.2)


#evaluating the model
print('[INFO] evaluating the model...')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy:', test_acc)
print('Test Loss:', test_loss)

#plotting 
#summarize training for Accuracy
print('[INFO] Plotting the Training Accuracy and loss')
plt.style.use('ggplot')
plt.figure()
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train','test'], loc = 'upper left')
plt.savefig('Mnist_Accuracy.png')
plt.show()

#summarize training for loss
plt.style.use('ggplot')
plt.figure()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train','test'],loc = 'upper left')
plt.savefig('Mnist_Loss.png')
plt.show()


