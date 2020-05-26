# **Behavioral Cloning** 

## Writeup/Readme

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use a simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia.png "Model Architecture"
[image2]: ./writeup_images/drive_example.jpg "Training"

---
### Code Overview

#### 1. File overview used to run the simulator in autonomous mode

This repo includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* drivingClone.ipynb contains the workflow I used when experimenting with different models to build the pipeline

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


#### 2. How to execute the model
Using a simulator from Udacity and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop a simple model to ensure that I could properly map the training data to steering output and then build up the model piece by piece to see which pieces helped or hindered the model and tune those pieces appropriately.

My first step was to use Keras to train a network to take an image from the center camera of the car as the input to the neural network and output a new steering angle for the car.

The network was used to verify that everything is working properly and consisted of a flattened image connected to a single output node. This single output node predicted the steering angle, which makes this a regression network and no activation function will be applied. This will try to minimize the error that the network predicts and the ground truth steering measurement. 

From there I built up the model to include normalization. Normalization is a way to remove effects of brightness or other image variations and focus on the content of the image instead of the brightness or other effects in the image. Note here, the simulator was not robust enough to verify that normalization was necessary - no apprant changing sun angles - but this step is crucial in the actual development of a self driving car. In this project, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py. 

Another feature the training data was that for track 1 the track was circular so the car was constantly turning to the left. This added a left turn bias to the car. To combat the left turn bias, I flipped all of the training data and measurements and added the flipped images and measurements to the training data. 

In a real car, we’ll have multiple cameras on the vehicle, and we’ll map recovery paths from each camera. For example, if you train the model to associate a given image from the center camera with a left turn, then you could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And you could train the model to associate the corresponding image from the right camera with an even harder left turn.

Multiple cameras were used in this project to feed in more images at different angles to the model. When recording the simulator saved images from three camers - center, left, and right - which were used to teach the model how to steer if the car drifts off to the left or the right. A steering offset was applied to add or subtract from the center angle for the left and right camera. The steering angle was determined based on experimentation. In a real self driving car - a full physics model would be used to determine the actual steering angle offset. 

A cropping layer was used to remove the top portion of the image and the hood of the car. The top portion of the image contained trees and other features that were not useful for training the model and the bottom portion of the image contained the hood of the car, which is fixed and can be removed. By adding in a cropping layer, new images coming from the simulator will be cropped when they are used on the final deep neural network. 

I initally started with the LeNet architecture to rapidly test measurments on a deep neural netowrk. This proved to be a good estimation for steering the car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that a dropout layer with 50% dropout was utilized.

Then I utilized the deep neural network for self driving cars published by NVIDIA for my architecture design and included the dropout layer, cropping layer, and normalization.  

Other steps in the design that can be found in the python notebook include visualizing the loss over multiple epochs and using generators. Generators were used to store less data in memory.

The final step was to run the simulator to see how well the car was driving around track one. From the image processing pipeline the driving clone was close to being able to run around the track, but saw issues before and after the bridge. Tuning the parameters such as epochs, dropout, and cropping along with including the augmented data solved these issues. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final model architecture

My model was adapted from the NVIDIA Deep Neural Network for self driving cars which can be seen below:

![alt text][image1]

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

A cropping layer was used to remove the top portion of the image and the hood of the car. The top portion of the image contained trees and other features that were not useful for training the model and the bottom portion of the image contained the hood of the car, which is fixed and can be removed. By adding in a cropping layer, new images coming from the simulator will be cropped when they are used on the final deep neural network. 

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers. Each convolution layer has a relu activation function.

After the five convolutional layers the model is flattened so fully connected layers can be operated on. After flattening, a dropout layer was added. To combat the overfitting, I modified the model so that a dropout layer with 50% dropout was utilized.

After the five convolutional layers, three fully connected layers are added, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but it should be noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.

The model used an adam optimizer, so the learning rate was not tuned manually and a mean square error was used as the loss function for the regression network. 


#### 3. Attempts to reduce overfitting in the model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that a dropout layer with 50% dropout was utilized.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 5. Appropriate training data

The sample driving data set was used as it proved to have good set of different cases to train the model.

Multiple cameras were used in this project to feed in more images at different angles to the model. When recording the simulator saved images from three camers - center, left, and right - which were used to teach the model how to steer if the car drifts off to the left or the right. A steering offset was applied to add or subtract from the center angle for the left and right camera. The steering angle was determined based on experimentation. In a real self driving car - a full physics model would be used to determine the actual steering angle offset. 

Another feature the training data was that for track 1 the track was circular so the car was constantly turning to the left. This added a left turn bias to the car. To combat the left turn bias, I flipped all of the training data and measurements and added the flipped images and measurements to the training data. 


#### 6. Creation of the Training Set & Training Process

The simulator was used to collect training data. Note here - that for this project the sample driving data set was used. To create additional data sets the following steps would be used:

* Enter Training Mode in the simulator.
* Start driving the car to get a feel for the controls.
* When you are ready, hit the record button in the top right to start recording.
* Continue driving for a few laps or till you feel like you have enough data.
* Hit the record button in the top right again to stop recording.

Strategies for Collecting Data

* the car should stay in the center of the road as much as possible
* if the car veers off to the side, it should recover back to center
* driving counter-clockwise can help the model generalize
* flipping the images is a quick way to augment the data
* collecting data from the second track can also help generalize the model
* we want to avoid overfitting or underfitting when training the model
* knowing when to stop collecting more data

The sample driving data contains 8037 images, which are summarized in `driving_log.csv`

the csv file contains a link to each frame along with the name of the associated center, left, right image file names, the steering angle, throttle value, brake value, and speed at each frame. The image below shows an example of the output from the center camera of the simulator. 

![alt text][image2]

I then preprocessed this data by normalizing and adding a cropping layer. Normalization is a way to remove effects of brightness or other image variations and focus on the content of the image instead of the brightness or other effects in the image. A cropping layer was used to remove the top portion of the image and the hood of the car. The top portion of the image contained trees and other features that were not useful for training the model and the bottom portion of the image contained the hood of the car, which is fixed and can be removed. By adding in a cropping layer, new images coming from the simulator will be cropped when they are used on the final deep neural network. 

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decreasing loss and validation loss value from the model. More than 5 epochs showed that the validation loss was increasing, which meant I was overfitting my training data. I used an adam optimizer so that manually training the learning rate wasn't necessary.

