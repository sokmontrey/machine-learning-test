import tensorflow as tf
import numpy as np

import os
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import matplotlib.pyplot as plt

"""
Dataset Path prep
"""
base_dir = "./datasets/"
train_dir = base_dir + "cat-vs-dog/train/"
validate_dir = base_dir + "cat-vs-dog/validate/"

train_cat_dir = train_dir + "Cat/"
train_dog_dir = train_dir + "Dog/"

validate_cat_dir = validate_dir + "Cat/"
validate_dog_dir = validate_dir + "Dog/"

num_train_cat = len(os.listdir(train_cat_dir))
num_train_dog = len(os.listdir(train_dog_dir))

num_validate_cat = len(os.listdir(validate_cat_dir))
num_validate_dog = len(os.listdir(validate_dog_dir))

print(base_dir)
print(train_dir)
print(validate_dir)

print(train_cat_dir)
print(train_dog_dir)

print("Num of train cats: ", num_train_cat)
print("Num of train dogs: ", num_train_dog)

print("Num of validate cats: ", num_validate_cat)
print("Num of validate dogs: ", num_validate_dog)

"""
Data prep
"""
BATCH_SIZE = 100
IMAGE_SIZE = 150

train_img_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    rotation_rage=45,
    width_shift=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)
validate_img_gen = ImageDataGenerator(rescale=1.0 / 255.0)

train_ds = train_img_gen.flow_from_directory(
    BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary",
)

validate_ds = validate_img_gen.flow_from_directory(
    BATCH_SIZE,
    directory=validate_dir,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary",
)
