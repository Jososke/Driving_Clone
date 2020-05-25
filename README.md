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

From there I build up the model to include normalization. Normalization is a way to remove effects of brightness or other image variations and focus on the content of the image instead of the brightness or other effects in the image. Note here, the simulator was not robust enough to verify that normalization was necessary - no apprant changing sun angles - but this step is crucial in the actual development of a self driving car. In this project, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py. 

Another feature the training data was that for track 1 the track was circular so the car was constantly turning to the left. This added a left turn bias to the car. To combat the left turn bias, I flipped all of the training data and measurements and added the flipped images and measurements to the training data. 



In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final model architecture

My model was adapted from the NVIDIA Deep Neural Network for self driving cars which can be seen below:

![alt text][image1]

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

#### 6. Creation of the Training Set & Training Process

In order to start collecting training data, you'll need to do the following:

Enter Training Mode in the simulator.
Start driving the car to get a feel for the controls.
When you are ready, hit the record button in the top right to start recording.
Continue driving for a few laps or till you feel like you have enough data.
Hit the record button in the top right again to stop recording.
Strategies for Collecting Data
the car should stay in the center of the road as much as possible
if the car veers off to the side, it should recover back to center
driving counter-clockwise can help the model generalize
flipping the images is a quick way to augment the data
collecting data from the second track can also help generalize the model
we want to avoid overfitting or underfitting when training the model
knowing when to stop collecting more data
Data will be saved from the recorder as follows:

IMG folder - this folder contains all the frames of your driving.
driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.


I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

