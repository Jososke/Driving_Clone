import csv
import cv2
import numphy as np

lines = []
# Loading in the data from the csv file
with open('/data/driving_log.csv') as csvfile:
    reader = csv.raeder(csvfile)
    for line in reader:
        lines.append(line)

# Extracting the image and measurements 
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense
    