import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np

''' Loads and preprocesses data. '''

# All letters

# train = pd.read_csv("./data/sign_mnist_train.csv")
# test = pd.read_csv("./data/sign_mnist_test.csv")

# labelBinarizer = LabelBinarizer()
# trainLabels = labelBinarizer.fit_transform(train["label"])
# testLabels = labelBinarizer.fit_transform(test["label"])

# trainAttributes = train.drop(labels = ["label"], axis=1)
# testAttributes = test.drop(labels = ["label"], axis=1)

# Filtered set

train = pd.read_csv("./data/sign_mnist_train.csv")
test = pd.read_csv("./data/sign_mnist_test.csv")

filtered_train = train[train['label'].isin([0, 1, 2])]
filtered_test = test[test['label'].isin([0, 1, 2])]

labelBinarizer = LabelBinarizer()
trainLabels = labelBinarizer.fit_transform(filtered_train["label"])
testLabels = labelBinarizer.fit_transform(filtered_test["label"])

trainAttributes = filtered_train.drop(labels=["label"], axis=1)
testAttributes = filtered_test.drop(labels=["label"], axis=1)

trainAttributes /= 255
testAttributes /= 255

trainAttributes = trainAttributes.values.reshape(-1, 28, 28, 1)
testAttributes = testAttributes.values.reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1, 
    width_shift_range=0.1,
    height_shift_range=0.1,
)

datagen.fit(trainAttributes)

''' 
    Building the CNN. 
    Using relu as activation function for entire CNN except output. 
    Using adam optimizer.
    Using categorical crossentropy loss function.
    Running for 10 epochs.
'''

model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(16, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(8, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
# model.add(Dense(units=24, activation="softmax"))
model.add(Dense(units=3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(datagen.flow(trainAttributes, trainLabels, batch_size=128), epochs=10, validation_data=(testAttributes, testLabels))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("models/relu/loss_plot.png")
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("models/relu/accuracy_plot.png")
plt.show()

predictions = model.predict(testAttributes)
predicted_labels = labelBinarizer.inverse_transform(predictions)
true_labels = labelBinarizer.inverse_transform(testLabels)
cm = confusion_matrix(true_labels, predicted_labels)

plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
classes = [str(i) for i in range(3)]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("models/relu/confusion_matrix.png")
plt.show()

model.save("models/relu/smnist.keras")
