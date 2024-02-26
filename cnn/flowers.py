import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
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
            (file_paths.reshape(5000, 1), labels.reshape(5000, 1))
            , axis=1)

np.random.shuffle(dataframe)

"""
Split dataset for train, validate, and test
"""
[train_f, rest_f] = np.split(dataframe, [int(0.8 * dataframe.shape[0])] )
[val_f, test_f] = np.split(rest_f, [int(0.5 * rest_f.shape[0])] )

print(train_f.shape)
print(val_f.shape)
print(test_f.shape)

# img_data_gen = ImageDataGenerator(rescale=1./255)
