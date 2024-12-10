import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import load_model

# Path to dataset
dataset_path = "vitiligo_renamed/datasets"

# Image parameters
img_height, img_width = 224, 224  # You can adjust this based on the GPU/compute capacity
batch_size = 32

# Prepare the data generator for loading images
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Update the number of output classes (from 6 to 7)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 classes for prediction, including 'leprosy'
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=1,  # You can increase the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save("skin_disease_model.h5")

# Optionally, you can also load the saved model later
# model = load_model("skin_disease_model.h5")

# To make predictions on a new image
def predict_skin_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)
    
    # Update class names to include 'leprosy'
    class_names = ['dermatomyositis', 'leprosy', 'morphea', 'normal', 'pityrasis_alba', 'psoriasis', 'vitiligo']
    predicted_class = class_names[class_index[0]]
    
    return predicted_class
