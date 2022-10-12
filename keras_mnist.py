
from sklearn.preprocessing import LabelBinarizer
# from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',required = True, help = 'path to the output loss/accuracy plot')

args = vars(ap.parse_args())

#Loading the mnist datasets
print('[INFO] loading MNIST dataset')
dataset = datasets.fetch_openml('mnist_784')
data = dataset.data.astype('float')/255.0
(x_train , x_test, y_train, y_test) = train_test_split(data, dataset.target, test_size = 0.25)

#converting intergers into vector representations
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


#define the architecture using keras
model = Sequential()
model.add(Dense(256, activation = 'sigmoid', input_shape = (784,)))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))

#train the model using SGD
print('[INFO] training the network...')
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics = ['accuracy'])
History = model.fit(x_train , y_train, validation_data = (x_test, y_test), epochs = 100,batch_size = 128)

#evaluating the model
print('[INFO] evaluating the network...')
predictions = model.predict(x_test, batch_size = 128)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))


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








 