import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras

# To stop potential randomness
seed = 128
# Neural Network for digit recognition

train_dir = "/home/sagar/IMAGR/Quiz/20_March_final/Data/Train"
test_dir = "/home/sagar/IMAGR/Quiz/20_March_final/Data/Test"
modelPath = "/home/sagar/IMAGR/Quiz/20_March_final/Data/model.h5";
modelJsonPath1   = "/home/sagar/IMAGR/Quiz/20_March_final/Data/modelJson.json"

# check for existence
os.path.exists(train_dir)
os.path.exists(test_dir)

trainData = []
trainLabel = []

listing = os.listdir(train_dir)
for img_name in listing:
    image_path = os.path.join(train_dir, img_name)
    label = img_name.split("_")[0]
    img = imread(image_path, flatten=True)
    trainData.append(img)
    trainLabel.append(label)

train_x = np.stack(trainData)
train_x /= 255.0
train_x = train_x.reshape(-1, 1024).astype('float32')

testData = []
testLabel = []
listing = os.listdir(test_dir)
for img_name in listing:
    image_path = os.path.join(test_dir, img_name)
    label = img_name.split("_")[0]
    img = imread(image_path, flatten=True)
    testData.append(img)
    testLabel.append(label)

test_x = np.stack(testData)
test_x /= 255.0
test_x = test_x.reshape(-1, 1024).astype('float32')
train_y = keras.utils.np_utils.to_categorical(testLabel)

#e
split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = trainLabel[:split_size], trainLabel[split_size:]

trainLabel[split_size:]


# define vars
input_num_units = 1024
hidden1_num_units = 50
hidden2_num_units = 50
hidden3_num_units = 50
hidden4_num_units = 50
hidden5_num_units = 50
output_num_units = 10

epochs = 5000
batch_size = 16800

# import keras modules

# create model
model = Sequential([
    Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
    Dropout(0.2),

    Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
])
# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

pred = model.predict_classes(test_x)

scores = model.evaluate(train_x, train_y, verbose=0)

print("Model Prediction Accuracy For Train Data: %.2f%%" % (scores[1] * 100))


scores1 = model.evaluate(test_x, testLabel, verbose=0)

print("Model Prediction Accuracy For Test Data: %.2f%%" % (scores1[1] * 100))

