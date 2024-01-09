# %% [markdown]
# Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %%
!pip install -q keras
!pip3 install torch torchvision
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
!pip install -q xgboost==0.4a30
!apt-get -qq install -y graphviz && pip install -q pydot

# %%
!pip install tensorflow[and-cuda]

# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# %%
import os

# %%
import os; os.system("xdg-open /media/hc17/I/SiameseNetwrok/aug1")

# %%
!pip install Pillow


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
                with Image.open(file_path) as img:
                    image_list.append(img)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return image_list

folder_path = "/media/hc17/I/SiameseNetwrok/aug1"
image_list = read_images_from_folder(folder_path)

# Now 'image_list' contains PIL Image objects of the images in the specified folder


# %%
import os
import uuid
from PIL import Image

def rename_images_with_uuid(folder_path):
    valid_extensions = {".jpg", ".jpeg"}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, extension = os.path.splitext(filename)

        if extension.lower() in valid_extensions:
            try:
                # Generate a unique name using UUID
                unique_name = str(uuid.uuid4()) + extension
                new_path = os.path.join(folder_path, unique_name)

                # Rename the file
                os.rename(file_path, new_path)

                print(f"Renamed {filename} to {unique_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Example usage
folder_path = "/media/hc17/I/SiameseNetwrok/aug1"
rename_images_with_uuid(folder_path)


# %%
import os
import uuid
from PIL import Image

def rename_images_with_uuid(folder_path):
    valid_extensions = {".jpg", ".jpeg"}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, extension = os.path.splitext(filename)

        if extension.lower() in valid_extensions:
            try:
                # Generate a unique name using UUID
                unique_name = str(uuid.uuid4()) + extension
                new_path = os.path.join(folder_path, unique_name)

                # Rename the file
                os.rename(file_path, new_path)

                print(f"Renamed {filename} to {unique_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Example usage
folder_path = "/media/hc17/I/SiameseNetwrok/aug2"
rename_images_with_uuid(folder_path)


# %%
import os
import uuid
from PIL import Image

def rename_images_with_uuid(folder_path):
    valid_extensions = {".jpg", ".jpeg"}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, extension = os.path.splitext(filename)

        if extension.lower() in valid_extensions:
            try:
                # Generate a unique name using UUID
                unique_name = str(uuid.uuid4()) + extension
                new_path = os.path.join(folder_path, unique_name)

                # Rename the file
                os.rename(file_path, new_path)

                print(f"Renamed {filename} to {unique_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Example usage
folder_path = "/media/hc17/I/SiameseNetwrok/aug3"
rename_images_with_uuid(folder_path)


# %%
# importing fucntional API's

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# %%
current_directory = os.getcwd()

# %%
current_directory

# %%
import os
import shutil

def merge_folders(source_folder1, source_folder2, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through the first source folder and copy images to the destination folder
    for filename in os.listdir(source_folder1):
        source_path = os.path.join(source_folder1, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)

    # Iterate through the second source folder and copy images to the destination folder
    for filename in os.listdir(source_folder2):
        source_path = os.path.join(source_folder2, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)

# Example usage:
source_folder1 = '/media/hc17/I/SiameseNetwrok/aug2'
source_folder2 = '/media/hc17/I/SiameseNetwrok/aug3'
destination_folder = '/media/hc17/I/SiameseNetwrok/negative_images'

merge_folders(source_folder1, source_folder2, destination_folder)


# %%


positive_path= os.path.join('GaugesData' , 'posGauges')
negative_path =  os.path.join('GaugesData' , 'negGauges')
anchor_path = os.path.join('GaugesData','ancGauges')






# %%
anchor = tf.data.Dataset.list_files(anchor_path + '/*.jpg').take(100)
positive = tf.data.Dataset.list_files(positive_path + '/*.jpg').take(100)

negative = tf.data.Dataset.list_files(negative_path + '/*.jpg').take(100)

# %%
test = positive.as_numpy_iterator()

# %%
test.next()

# %%
# preprocessing

# %%
def preprocess(filePath):
    byteImage = tf.io.read_file(filePath)
    img = tf.io.decode_jpeg(byteImage)
    img = tf.image.resize(img, (100,100))
    img = img/255.0
    return img

# %%
trialPositiveImages = preprocess('GaugesData/posGauges/0e4a6860-e18f-477e-a3f3-fdc60bf21c59.jpg')

# %%
plt.imshow(trialPositiveImages)

# %%
# labelled data

# %%
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)### all data whether positive or negative gets concatenated

# %%
sampleData = data.as_numpy_iterator()

# %%
example = sampleData.next()

# %%
example

# %%
# train and test partition

# %%
def preprocess_twin(inputImage , validationImage, label):
    return(preprocess(inputImage),preprocess(validationImage), label)

# %%
res = preprocess_twin(*example)
# *example passed because

# %%
res

# %%
len(res)

# %%
res[0]

# %%
plt.imshow(res[0])

# %%
plt.imshow(res[1])

# %%
res[2]

# %%
# data loader pipeline?

# %%
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size = 1024)

# %%

samps = data.as_numpy_iterator()

# %%
sample = samps.next()

# %%
plt.imshow(sample[0])

# %%
plt.imshow(sample[1])

# %%
sample[2]

# %%
len(data)

# %%
training_data = data.take(round(len(data) * .7))
# passing data in the form of batches instead all at once
training_data = training_data.batch(5)
# prefetch the upcoming image so to avoid any kind of bottleneck in the NN
training_data = training_data.prefetch(7)

# %%
train_samples = training_data.as_numpy_iterator()

# %%
train_samples = train_samples.next()

# %%
train_samples

# %%
len(train_samples)

# %%
len(train_samples[0])

# %%
#testing partitions

# %%
test_data = data.skip(round(len(data) * .7)) # do not take the 70 percent data
test_data = test_data.take(round(len(data) * .3)) # take the remaining 30 percent data

# %% [markdown]
# Model Engineering

# %%
def makeEmbedding():
    inp = Input(shape=(100,100,3), name = 'input_image')


    #first block - 
        # convolution layer1
        # max pooling layer 1
    
    convolutionLayer1 = Conv2D(64,(10,10), activation ='relu')(inp)  ## the code is pertaining to the reference research paper on one shot 
    maxPoolingLayer1 = MaxPooling2D(64,(2,2), padding='same')(convolutionLayer1)

    # second block:
        # convolution layer 2
        # max pooling layer 2
    convolutionLayer2 = Conv2D(128,(7,7), activation='relu')(maxPoolingLayer1)
    maxPoolingLayer2 = MaxPooling2D(64,(2,2), padding='same')(convolutionLayer2)

    # Third block
        # convolution layer 3
        # convolution layer 3
    convolutionLayer3 = Conv2D(128,(4,4), activation='relu')(maxPoolingLayer2)
    maxPoolingLayer3 = MaxPooling2D(64,(2,2), padding='same')(convolutionLayer3)

    # Final Block
        # Convolution Layer 4
        # Flattening to produce Feature Vector

    convolutionLayer4 = Conv2D(246,(4,4), activation= 'relu')(maxPoolingLayer3)
    flattenLayer1 = Flatten()(convolutionLayer4)
    #dense layer
    denseLayer1 = Dense(4096, activation='sigmoid')(flattenLayer1)

    return Model(inputs=[inp],outputs=[denseLayer1],name='embedding')

# %%
embedding = makeEmbedding()

# %%
embedding.summary()

# %%
# Distance Layer Build

# %%
# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# %% [markdown]
# Siamese Model

# %%
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# %%
# load pretrained model
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# %%


def makeSiameseModel():
    # Inputs for anchor and validation images
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Feature extraction using VGG-16
    vgg16_embedding = vgg16_base(input_image)
    vgg16_validation_embedding = vgg16_base(validation_image)

    # Flatten the VGG-16 embeddings
    flatten_layer = Flatten()

    # Flatten VGG-16 embeddings for anchor and validation images
    anchor_embedding = flatten_layer(vgg16_embedding)
    validation_embedding = flatten_layer(vgg16_validation_embedding)

    # Combine Siamese Distance Components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(anchor_embedding, validation_embedding)

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='Siamese-Network')



# %%
# Create the Siamese model
siamese_model = makeSiameseModel()

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])

# Print the model summary
siamese_model.summary()

# Train the Siamese model
EPOCHS = 50


# %%
train(training_data, EPOCHS)

# %%


# %%
siamese_model.summary()

# %%
# Data training

# %%
binaryCrossLoss = tf.losses.BinaryCrossentropy()

# %%
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

# %%

checkpoint_dir = './training_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# %%
@tf.function
# @tf allows us to compile a function into a callable Tensor flow graph
def train_step(batch):
    with tf.GradientTape() as tape:

        #Get anchor and pos/neg images:
        X = batch[:2]
        # Get labels for the images
        Y = batch[2]

        # Forward pass:
        Yhat = siamese_model(X, training = True)

        # Calculate Loss:
        loss = binaryCrossLoss(Y , Yhat)


    #Calculate gradients:
    grad = tape.gradient(loss , siamese_model.trainable_variables)

    # calculate updated weights and apply them to the model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

# %%
from tensorflow.keras.metrics import Recall, Precision

# %%
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating metric objects 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


# %%
EPOCHS = 50

# %%
train(training_data, EPOCHS)

# %%



