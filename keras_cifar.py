#importing required packages
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import classification_report
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import cifar10
import tensorflow 
import matplotlib.pyplot as plt
import numpy as np
import argparse

#construct the argumnet parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',required = True, help = '#path to the output loss/accuracy plotting')
args = vars(ap.parse_args())

#loading the CIFAR-10 dataset
#it consist total 60,000 images of 10 different classes that why its name cifar-10 because its having 10 output classes
#out of 60,000images. 50,000images use for training and rest for testing.
print('[INFO] loading CIFAR 10 Dataset...')
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#converting its data type and scaling it into range [0,1]
x_train = x_train.astype('float')/255.0
x_test = x_test.astype('float')/255.0

#Reshaping the matrix   (32*32*3 = 3072, (each image is of 32*32 height n width & 3 is channel RGB) flatten into single list of floating point)
x_train = x_train.reshape(x_train.shape[0],3072)
x_test = x_test.reshape(x_test.shape[0],3072)

#converts the labels from integers to vectors
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

labelNames = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Network architecture

model = Sequential()
model.add(Dense(1024 , input_shape = (3072,), activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#train the model using sgd
sgd = SGD(0.01, momentum = 0.5)
model.compile(loss = 'categorical_crossentropy',optimizer = sgd, metrics = ['accuracy'])
History = model.fit(x_train , y_train , batch_size = 32, epochs = 100, validation_data = (x_test, y_test))

#evaluate the network
print('[INFO] evaluating the network...')
predictions = model.predict(x_test, batch_size = 32)
print(predictions)
print(classification_report(y_test.argmax(axis = 1), predictions.argmax(axis=1), target_names = labelNames))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Testing_Accuracy :', test_acc)
print('Testing loss :', test_loss)

#plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), History.history['loss'], label = 'train_loss')
plt.plot(np.arange(0,100), History.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0,100), History.history['accuracy'], label= 'train_acc')
plt.plot(np.arange(0,100), History.history['val_accuracy'], label = 'val_acc')
plt.title('Training loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['output'])


# sgd 0.01 epoch 10
# Testing_Accuracy : 0.47189998626708984
# Testing loss : 1.4704790115356445

# sgd 0.01 momentum 0.5 epoch 10
#Testing_Accuracy : 0.515999972820282
# Testing loss : 1.3731591701507568
  
#  gd 0.01 momentum 0.5 epoch 100
# Testing_Accuracy : 0.5764999985694885
# Testing loss : 2.8420872688293457