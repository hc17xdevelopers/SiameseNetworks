# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image

# %%
from PIL import Image
import os

def read_images_from_folder(folder_path):
    image_list = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}  # Add more if needed

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, extension = os.path.splitext(filename)

        if extension.lower() in valid_extensions:
            try:
                with Image.open(file_path).convert("RGB") as img:
                    image_list.append(img)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return image_list


# %%
g

# %%



# %%
# Function to preprocess images
def preprocess_image(image):
    img = image.resize((100, 100))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to create positive, anchor, and negative datasets for a specific class
def create_datasets_for_class(class_name, data_folder):
    anchor_images = []
    positive_images = []
    negative_images = []

    class_folder = os.path.join(data_folder, class_name)
    images = read_images_from_folder(class_folder)

    # Split images into anchor, positive, and negative
    anchor_images += images[:50]
    positive_images += images[50:]
    for other_class in os.listdir(data_folder):
        if other_class != class_name:
            other_class_folder = os.path.join(data_folder, other_class)
            negative_images += read_images_from_folder(other_class_folder)

    return anchor_images, positive_images, negative_images


# %%
# Define the Siamese network architecture
class SiameseNetwork(tf.keras.Model):
    def __init__(self, num_classes):
        super(SiameseNetwork, self).__init__()

        # Define the shared layers
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')  # Output layer for multiple classes

    def call(self, inputs):
        # Extract the two images from the input
        image1, image2 = inputs

        # Pass the images through the shared layers
        shared_output1 = self.fc1(image1)
        shared_output2 = self.fc1(image2)

        # Pass the flattened outputs through the fully connected layers
        fc1_output1 = self.fc2(shared_output1)
        fc1_output2 = self.fc2(shared_output2)

        fc2_output1 = self.fc3(fc1_output1)
        fc2_output2 = self.fc3(fc1_output2)

        return fc2_output1, fc2_output2

# %%
# Training function for each class
def train_for_class(class_name, data_folder, num_epochs=10):
    anchor_images, positive_images, negative_images = create_datasets_for_class(class_name, data_folder)

    anchor_data = np.array([preprocess_image(img) for img in anchor_images])
    positive_data = np.array([preprocess_image(img) for img in positive_images])
    negative_data = np.array([preprocess_image(img) for img in negative_images])

    num_classes = 3  # Adjust based on the actual number of classes
    anchor_labels = tf.one_hot(np.zeros(len(anchor_data)), num_classes)
    positive_labels = tf.one_hot(np.zeros(len(positive_data)), num_classes)
    negative_labels = tf.one_hot(np.ones(len(negative_data)), num_classes)

    siamese_network = SiameseNetwork(num_classes)
    optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed
    siamese_network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nTraining for class: {class_name}, Epoch: {epoch + 1}/{num_epochs}")
        siamese_network.fit([anchor_data, positive_data], anchor_labels, epochs=1, verbose=2)

    return siamese_network

# %%
# Train the model for each class
classes = ['aug1', 'aug2', 'aug3']  # Update with your class names
data_folder = '/media/hc17/I/SiameseNetwrok/multidataset'  # Update with your data folder
models = {}

for class_name in classes:
    model = train_for_class(class_name, data_folder, num_epochs=3)
    models[class_name] = model

# %%
# Function to predict the class of an input image
def predict_class(input_image, models):
    input_data = np.array([preprocess_image(input_image)])

    class_scores = {}
    for class_name, model in models.items():
        score = model.predict([input_data, input_data])
        class_scores[class_name] = score[0]

    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class

# Example usage:
input_image_path = '/path/to/your/input/image.jpg'  # Update with the path to your input image
input_image = Image.open(input_image_path)
predicted_class = predict_class(input_image, models)
print(f"Predicted class: {predicted_class}")


