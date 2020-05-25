import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D

lines = []
# Loading in the data from the csv file
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# create adjusted steering measurements for the side camera images
correction = 0.2 # this is a parameter to tune

# Extracting the image and measurements 
images = []
measurements = []
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        imageBGR = cv2.imread(current_path)
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
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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

# Using mean square error for regression network
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

# Saving to file for use in drive.py
model.save('model.h5')