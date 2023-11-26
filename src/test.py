import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import numpy as np
import string

train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

trainLabels = train["label"]
testLabels = test["label"]

trainAttributes = train.values
testAttributes = test.values

num_images_to_display = 100
success = []

model = load_model("models/relu/smnist.keras")

label_mapping = {idx: letter for idx, letter in enumerate(string.ascii_uppercase) if letter not in ['J', 'Z']}

for i in range(num_images_to_display):

    image = trainAttributes[i][1:].reshape((28, 28))

    image = pd.DataFrame(image.flatten()).T
    image = image.values.astype(float) / 255.0
    image = image.reshape(-1, 28, 28, 1)

    predicted_sign = model.predict(image)
    top_label = np.argsort(predicted_sign, axis=1)[0][-1:][::-1]
    label = trainLabels[i]
    top_label = int(top_label[0])

    if label == top_label:
        success.append(1)
    else:
        success.append(0)

print(np.mean(success))

