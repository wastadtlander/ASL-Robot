import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("./data/sign_mnist_train.csv")
test = pd.read_csv("./data/sign_mnist_test.csv")

trainLabels = train["label"]
testLabels = test["label"]

trainAttributes = train.values
testAttributes = test.values

num_images_to_display = 5

fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 3))  # Creating subplots

for i in range(num_images_to_display):
    image = trainAttributes[i][1:].reshape((28, 28))
    label = trainLabels[i]

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout() 
plt.show()