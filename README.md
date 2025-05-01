# Sign Language Detection

This project is a real-time sign language alphabet recognizer built using **OpenCV** and **Keras**. It currently detects letters from **A to Z**, but can be extended to recognize a larger set of signs, including the numbers.

## 🔧 Prerequisites

To start this project, you need to install the following software and libraries:

- **Python 3.7.4**
- **Jupyter Notebook** or any Python IDE
- **NumPy 1.16.5**
- **OpenCV 3.4.2** (`cv2`)
- **Keras 2.3.1**
- **TensorFlow 2.0.0**

## 📁 Project Structure

The project structure is organized as follows:

```
SignLanguageRecognition/
│
├── create_gesture_data.py   # For dataset generation
├── train_cnn_model.py       # For training the CNN
├── model_for_gesture.py     # For real-time prediction
├── gesture/
│   ├── train/
│   │   ├── a/
│   │   ├── b/
│   │   └── ...
│   └── test/
│       ├── a/
│       ├── b/
│       └── ...
```

## 1. Creating the Dataset

To create the dataset, follow these steps:

- **Live Camera Feed**: Use a live camera feed to capture hand gestures.
- **ROI (Region of Interest)**: Create an ROI in each frame.
- **Background Learning**: Learn the background over the first 60 frames using an accumulated weighted average.
- **Hand Segmentation**: Segment the hand from the background using contour detection.
- **Image Storage**: Store images in folders corresponding to each letter (A–Z) for both train and test sets.

Here is an example of setting the ROI box dimensions:

```
# ROI box dimensions
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350
```

## 2. Training the CNN

### Loading Data

Use `ImageDataGenerator` from Keras to load the data:

```
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory='gesture/train', target_size=(64,64), class_mode='categorical', batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory='gesture/test', target_size=(64,64), class_mode='categorical', batch_size=10)
```

### CNN Architecture

Define the CNN model architecture:

```
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### Training

Compile and train the model:

```
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(train_batches, epochs=10, validation_data=test_batches, callbacks=[reduce_lr, early_stop])
model.save('best_model_dataflair3.h5')
```

## 🤖 3. Real-Time Gesture Prediction

For real-time prediction:

- **Live Webcam Feed**: Process the live webcam feed.
- **ROI Isolation**: Isolate the ROI.
- **Hand Segmentation**: Segment the hand from the background.
- **Prediction**: Pass the preprocessed image to the trained CNN model for prediction.

Here’s how to load the model and predict:

```
from keras.models import load_model

model = load_model('best_model_dataflair3.h5')

```

## 🧪 Testing and Evaluation

Evaluate the model:

```
scores = model.evaluate(imgs, labels, verbose=0)
print(f'Loss: {scores} | Accuracy: {scores*100:.2f}%')
```

## 🌟 Final Notes

- This system can be extended to recognize digits, custom gestures, or even full sign language sentences.
- Data quality and lighting conditions significantly affect accuracy.
```

