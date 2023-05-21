# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

#1. Setup Dataset
dataset_dir = '/content/drive/MyDrive/ML'   # Define the path to your dataset directory in Google Drive

img_width, img_height = 180, 180            # Define the image size

num_classes = len(os.listdir(dataset_dir))  # Define the number of classes (Celebrity classes)

def load_drive_dataset():
    data, labels = [], []
    for celebrity_dir in os.listdir(dataset_dir):
        celebrity_path = os.path.join(dataset_dir, celebrity_dir)
        if os.path.isdir(celebrity_path):
            images = os.listdir(celebrity_path)
            for image in images:
                image_path = os.path.join(celebrity_path, image)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (img_width, img_height))
                data.append(img)
                labels.append(celebrity_dir)
    return data, labels

data, labels = load_drive_dataset()    # Load the dataset

#2. Split the dataset to trainging and testing data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

#3. Preprocessing the data (Normalization) standardize values to be in the [0, 1] range using numpy
train_data = np.array(train_data) / 255.0
test_data = np.array(test_data) / 255.0

# Convert labels to integer value then to matrix (categorical)
# If integer value 1 is placed in the column corresponding to the class label, while the other columns contain 0.
# Images are the rows, & labels are the columns
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels) # The original labels are replaced with their corresponding integer values.
test_labels = label_encoder.transform(test_labels)
train_labels = to_categorical(train_labels)              # Convert the integer-encoded labels into matrix (categorical) format.
test_labels = to_categorical(test_labels)

#4. Define first Model
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)), # First hidden layer, 32 number of neurons in each hidden layer
    MaxPooling2D((2, 2)),    # It effectively reduces the size of the image while retaining the most prominent features. 
    Flatten(),  # Converting the 2D representation to a 1D representation. This is necessary to connect it to the subsequent fully connected layers.
    Dense(64, activation='relu'), # This is a fully connected layer with 64 neurons and a sigmoid activation function. It takes the flattened output as input.
    Dense(num_classes, activation='softmax') # It uses the sigmoid activation function to generate the output probabilities for each class.
])
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Prepare the model for training, The loss function measures the diffrenece between the predicted output and the actual output

# Train the first model
model1.fit(train_data, train_labels, epochs=7, batch_size=32, validation_data=(test_data, test_labels)) # Epochs is the number of times the entire training dataset is used for learning.

#5. Evaluate the first model
loss, accuracy1 = model1.evaluate(test_data, test_labels)
print("")
print("Model 1 Accuracy:", accuracy1)
print("")

# Define Second Model
model2 = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),  # First hidden layer, 64 number of neurons in each hidden layer
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Second hidden layer, 128 number of neurons in each hidden layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the second model
model2.fit(train_data, train_labels, epochs=7, batch_size=32, validation_data=(test_data, test_labels))

# Evaluate the second model
loss, accuracy2 = model2.evaluate(test_data, test_labels)
print("")
print("Model 2 Accuracy:", accuracy2)
print("")

# Show 10 random results from the testing set.
random_indices = random.sample(range(len(test_data)), 10)

print("Random 10 Results:")

for index in random_indices:
    image = test_data[index]

    # Reshape the image for prediction
    image = np.expand_dims(image, axis=0)

    # Get the image class name
    image_name = label_encoder.inverse_transform([np.argmax(test_labels[index])])

    # Make predictions
    prediction1 = model1.predict(image)
    prediction2 = model2.predict(image)

    # Get the predicted class name
    predicted_class1 = label_encoder.inverse_transform([np.argmax(prediction1)])
    predicted_class2 = label_encoder.inverse_transform([np.argmax(prediction2)])

    print("Image to be predicted:", image_name[0])
    print("Model 1 Prediction:", predicted_class1[0])
    print("Model 2 Prediction:", predicted_class2[0])
    print("-------------------------------------------")
    print("")
    
    
    
    
    # ____________________________________________________________________________________________________________________________________________

              # Celebrating handing over my last assignment as undergarduate of Software Engineering at Cairo University ! 🎉🎓

                          # "Celebrate endings, for they precede new beginnings." - Jonathan Lockwood Huie