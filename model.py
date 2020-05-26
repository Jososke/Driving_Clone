import os
import csv
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
import sklearn
from sklearn.utils import shuffle

samples1 = []
samples2 = []
samples = []
# Loading in the data from the csv file

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples1.append(line)

with open('data/data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples2.append(line)

samples = samples1[1:] + samples2

# create adjusted steering measurements for the side camera images
correction = 0.2 # this is a parameter to tune

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    imageBGR = cv2.imread('data/IMG/' + filename)
                    if imageBGR is None:
                        imageBGR = cv2.imread('data/data2/IMG/' + filename)
                    # Images in drive.py are read in as RGB
                    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    if i == 0: # Center
                        measurement = float(line[3])
                    elif i == 1: # Left
                        measurement = float(line[3]) + correction
                    elif i == 2: # Right
                        measurement = float(line[3]) - correction        
                    measurements.append(measurement)

            # Augmenting data to avoid left bias on the track
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
                
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
# set up lambda layer for normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# cropping 70 pixels from top of image (trees) and 25 pixels from bottom of image (hood of car)
model.add(Cropping2D(cropping=((70,25),(0,0))))

# NVIDIA architecture and including a dropout layer for redundancy
model.add(Conv2D(24,5,5, subsample = (2,2), activation = "relu"))
model.add(Conv2D(36,5,5, subsample = (2,2), activation = "relu"))
model.add(Conv2D(48,5,5, subsample = (2,2), activation = "relu"))
model.add(Conv2D(64,3,3, activation = "relu"))
model.add(Conv2D(64,3,3, activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.5))   
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=np.ceil(len(validation_samples)/batch_size), 
                    epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save the model
model.save('model.h5')