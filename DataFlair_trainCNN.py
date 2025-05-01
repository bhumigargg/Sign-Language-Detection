import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import itertools
import random
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

# Base directory containing all letter directories
base_dir = r"E:\\code\\output_images"

# Directories for train and test datasets
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split dataset into train and test
for letter in os.listdir(base_dir):
    letter_path = os.path.join(base_dir, letter)
    
    # Skip if it's not a directory (e.g., train/test folders themselves)
    if not os.path.isdir(letter_path) or letter in ["train", "test"]:
        continue

    # Get all image file paths for the current letter
    images = [os.path.join(letter_path, img) for img in os.listdir(letter_path) if img.endswith(".jpg")]

    # Split images into train (80%) and test (20%) sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create subdirectories for the current letter in train and test directories
    train_letter_dir = os.path.join(train_dir, letter)
    test_letter_dir = os.path.join(test_dir, letter)
    os.makedirs(train_letter_dir, exist_ok=True)
    os.makedirs(test_letter_dir, exist_ok=True)

    # Move train images to the train directory
    for img in train_images:
        shutil.copy(img, os.path.join(train_letter_dir, os.path.basename(img)))

    # Move test images to the test directory
    for img in test_images:
        shutil.copy(img, os.path.join(test_letter_dir, os.path.basename(img)))

print("Dataset split into train and test directories successfully!")

# Train and test paths
train_path = train_dir
test_path = test_dir

# Load train and test batches
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

# Get a batch of images and labels from the training dataset
imgs, labels = next(train_batches)

# Define the function to plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))  # 1 row, 10 columns
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        ax.imshow(img)
        ax.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()

# Plot the images
plotImages(imgs)

# Print the shape of the images and labels
print("Images shape:", imgs.shape)
print("Labels:", labels)

# Define the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))  # Assuming 26 classes for letters a-z

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Train the model
history = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)

# Save the model
model.save('best_model_letters.h5')

print("Model training complete and saved as 'best_model_letters.h5'.")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_batches, verbose=1)
print("Test Accuracy:", test_accuracy)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()