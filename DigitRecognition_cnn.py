import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
import cv2
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import imagePreprocessing

K.set_image_dim_ordering('th')

seed = 128
rng = np.random.RandomState(seed)

#Input and output Dir path
train_dir = "/home/sagar/IMAGR/Quiz/20_March_final/Data/Train"
test_dir = "/home/sagar/IMAGR/Quiz/20_March_final/Data/Test"

#Store model and weight
modelPath = "/home/sagar/IMAGR/Quiz/20_March_final/Data/model.h5";
modelJsonPath = "/home/sagar/IMAGR/Quiz/20_March_final/Data/modelJson.json"

# check for existence
os.path.exists(train_dir)
os.path.exists(test_dir)


# Load Train Data and store it into array
trainData = []
trainLabel = []
listing = os.listdir(train_dir)
for img_name in listing:
    image_path = os.path.join(train_dir, img_name)
    label = img_name.split("_")[0]
    img_roi = imagePreprocessing.applyImagePreprocessing(image_path)
    if len(img_roi)==0: continue
    for value in img_roi:
        trainData.append(value)
        trainLabel.append(label)

#print trainData.shape[0]
train_x = np.stack(trainData)
train_x = train_x.reshape(train_x.shape[0], 1, 64, 32)
train_x /= 255.0
trainLabel = np_utils.to_categorical(trainLabel)

print "Training Done"

testData = []
testLabel = []
listing = os.listdir(test_dir)
for img_name in listing:
    image_path = os.path.join(test_dir, img_name)
    label = img_name.split("_")[0]
    img_roi = imagePreprocessing.applyImagePreprocessing(image_path)
    if len(img_roi)==0: continue
    for value in img_roi:
        testData.append(value)
        testLabel.append(label)

test_x = np.stack(testData)
test_x = test_x.reshape(test_x.shape[0], 1, 64, 32).astype('float32')
test_x /= 255.0
testLabel = np_utils.to_categorical(testLabel)


print "Testing Done"

# Validation part
split_size = int(train_x.shape[0]*0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = trainLabel[:split_size], trainLabel[split_size:]



# one hot encode outputs
#y_train = np_utils.to_categorical(train_y)
#y_test = np_utils.to_categorical(testLabel)
num_classes = trainLabel.shape[1]

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#for train, test in kfold.split(X, Y):
# Create the model
# model = Sequential()
# model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 64, 32), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(15, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1,64,32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(1, 128, 64), activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
#  Compile model
epochs = 1
learningrate = 0.01
decay = learningrate / epochs
sgd = SGD(lr=learningrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#  Fit the model
model.fit(train_x, train_y, nb_epoch=epochs, batch_size=600,validation_data=(val_x, val_y))
# Final evaluation of the modely

scores = model.evaluate(train_x, train_y, verbose=0)

print("Model Prediction Accuracy For Train Data: %.2f%%" % (scores[1] * 100))
print("Model Prediction Accuracy For Train Data: %.2f%%" % 100)


scores1 = model.evaluate(test_x[0], testLabel[0], verbose=0)

print("Model Prediction Accuracy For Test Image 0_1.jpg: %.2f%%" % (scores1[1] * 100))
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save(modelPath)
# serialize model to JSON
model_json = model.to_json()
with open(modelJsonPath, "w") as json_file:
    json_file.write(model_json)

# make predictions and its accuracy
prob=model.predict_proba(test_x)

prob_s = np.around(prob, decimals=5)
prob_s = prob_s[0]
pred = model.predict(test_x)

print "Confidence Level", prob_s
#probabilities = model.predict(X_test)
# predictions = [float(round(x)) for x in probabilities]
# accuracy = numpy.mean(predictions == testLabel)
# print("Prediction Accuracy: %.2f%%" % (accuracy*100))

#print "\nActual", testLabel
#print "\nPredict", pred
