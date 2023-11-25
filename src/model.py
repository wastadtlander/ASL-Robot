import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

''' Loads and preprocesses data. '''

train = pd.read_csv("./data/sign_mnist_train.csv")
test = pd.read_csv("./data/sign_mnist_test.csv")

labelBinarizer = LabelBinarizer()
trainLabels = labelBinarizer.fit_transform(train["label"])
testLabels = labelBinarizer.fit_transform(test["label"])

trainAttributes = (train.values[:, 1:] / 255).reshape(-1, 28, 28, 1)
testAttributes = (test.values[:, 1:] / 255).reshape(-1, 28, 28, 1)

''' 
    Building the CNN. 
    Using relu as activation function for entire CNN except output. 
    Using adam optimizer.
    Using categorical crossentropy loss function.
    Running for 10 epochs.
'''

# TODO: ^ Check if other activation functions/optimizer/loss/epochs work better (doubt it will).
# TODO: Play around with and eliminate layers that are unneeded.

model = Sequential()
model.add(Conv2D(75, (3,3), strides = 1, padding = "same", activation = "relu", input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = "same"))
model.add(Conv2D(50, (3,3), strides = 1, padding = "same", activation = "relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = "same"))
model.add(Conv2D(25, (3,3), strides = 1, padding = "same", activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = "same"))
model.add(Flatten())
model.add(Dense(units = 512, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(units = 24, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()

history = model.fit(trainAttributes, trainLabels, batch_size = 128, epochs = 10, validation_data = (testAttributes, testLabels))

model.save("models/relu/smnist.keras")
