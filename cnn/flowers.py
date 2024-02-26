import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras import layers
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os

datasets_path = "datasets/flower_images"

"""
Create list of datasets paths and list of labels
"""
all_class_names = os.listdir(datasets_path)
file_paths = []
labels = []

for class_name in all_class_names:
    class_path = os.path.join(datasets_path, class_name)
    all_image_names = os.listdir(class_path)

    labels.append([class_name] * len(all_image_names))

    for image_name in all_image_names:
        image_path = os.path.join(class_path, image_name)

        file_paths.append(image_path)

"""
Combine paths and corresponding labels together
"""
file_paths = np.array(file_paths)
labels = np.array(labels).flatten()
dataframe = np.concatenate(
    (file_paths.reshape(5000, 1), labels.reshape(5000, 1)), axis=1
)

np.random.shuffle(dataframe)

"""
Split dataset for train, validate, and test
"""
[train_df, rest_df] = np.split(dataframe, [int(0.8 * dataframe.shape[0])])
[val_df, test_df] = np.split(rest_df, [int(0.5 * rest_df.shape[0])])

train_gen = ImageDataGenerator(rescale=1.0 / 255)
val_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)

input_size = (32, 32)

train_ds = train_gen.flow_from_dataframe(
    pd.DataFrame(train_df, columns=["image_path", "label"]),
    x_col="image_path",
    y_col="label",
    target_size=input_size,
    class_mode="categorical",
)

val_ds = val_gen.flow_from_dataframe(
    pd.DataFrame(val_df, columns=["image_path", "label"]),
    x_col="image_path",
    y_col="label",
    target_size=input_size,
    class_mode="categorical",
)

test_ds = test_gen.flow_from_dataframe(
    pd.DataFrame(test_df, columns=["image_path", "label"]),
    x_col="image_path",
    y_col="label",
    target_size=input_size,
    class_mode="categorical",
)

model = Sequential()
model.add(
    layers.Conv2D(
        32, (5, 5), activation="relu", input_shape=(input_size[0], input_size[1], 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))

model.compile(
    optimizer="adam",
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds, batch_size=32, epochs=20, validation_data=val_ds, verbose=2
)

print("Completed training")

model.save('./cnn/flowers.h5')

print("Saved model")

final_eval = model.evaluate_generator(test_ds)
print("Test Loss: ", final_eval[0])
print("Test Accuracy: ", final_eval[1])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./cnn/flowers_history.png")
plt.show()
