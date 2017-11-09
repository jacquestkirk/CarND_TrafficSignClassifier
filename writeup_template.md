# Traffic Sign Recognition



---

## Goals

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[imageFrequency]: ./WriteupImages/frequency.png
[imageExamples]: ./WriteupImages/signs.png
[accuracyVsTime]: ./Plots/Round4/sweep%20batchSize_%20round%204_accuracyVsTime.png
[accuracyVsEpoch]: ./Plots/Round4/sweep%20batchSize_%20round%204.png
[webSign0]: ./test_images/scaled/50kph.jpg
[webSign1]: ./test_images/scaled/120kph.jpg
[webSign2]: ./test_images/scaled/BumpyRoad.jpg
[webSign3]: ./test_images/scaled/end_no_passing.jpg
[webSign4]: ./test_images/scaled/RightTurn.jpg


---
## Summary

* training set accuracy of **99.7**
* validation set accuracy of **96.3**
* test set accuracy of **96.3**

---

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jacquestkirk/CarND_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

First, I printed an image of each sign to see what I'm working with. 

![imageExamples]

Then I created a chart of the number of each sign reperesented in the training set. 

![imageFrequency]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1. **Grayscale:** I originally converted to grayscale. Converting to grayscale reduces the depth of the input set by a factor of 3, reducing neural network complexity. 
	
	However with grayscale I struggled to get the required 93% accuracy, so I reverted back to color images to give the neural net more information to train on. 

2. **Normalization:** I normalized the data by bringing the data closer to zero mean and restricting image values between -0.5 and 0.5. 
	
	Zero mean helps to take advantage of my relu activation function since the non-linearity occurs at 0. In addition, it reduces the number of steps gradient descent needs to get into the right vicinity. 
	
	Scaling so that pixel values are between -0.5 and 0.5 ensures that calculated outputs don't explode such that we saturate or swamp out numbers when we sum. 

3. **Augmentation:** I did not augment the data, mostly because I saw good enough performance and I was too lazy to do this. However, as a result of this, some of the trickier traffic sign cases I found on the web are not detected well by my neural network. I will discuss this in a later section. 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with the architecture from the Lenet lab. I found that it was overfitting, so I added dropout layers. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|-----------------------|-----------------------------------------------| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Divide by # weights 	| To keep output magnitude in check 			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x25 	|
| Divide by # weights 	| To keep output magnitude in check 			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x25 	   				|
| Fully connected		| length 240        							|
| Divide by # weights 	| To keep output magnitude in check 			|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| length 62        								|
| Divide by # weights 	| To keep output magnitude in check 			|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Output				| length 43        								|
| Softmax				| 			 									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters:

* learning rate: 0.02
* batch size: 128
* epochs: 20
* keep probablity: 50% (for both layers)
* loss function: mean softmax cross entropy
* optimizer: adam optimizer

I swept learning rate, batch size, and keep probability to find the optimum values for them. Number of epochs was chosen by looking at when the acuracy started to plateau. Loss function and optimizer were copied from the functional Lenet lab. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99.7**
* validation set accuracy of **96.3**
* test set accuracy of **96.3**

I followed a windy path to get to the network I have now. Parts of it were iterative, and other parts were deterministic. For now I'll abstract away the iteration, but I'll return to the specifics of it further down. 

Overall flow:

1. First I started with the Lenet lab. I chose it because it was the only architecture I knew at the time. Converted images to grayscale and pushed it through using the same hyperparameters. 
2. I tried to make this work, so I iterated over hyperameters as described in the section below. 
3. Despite numerates iterations, I could not get the desired 93% (got stuck at something like 91%). I decided to remove the grayscale step so that the network would have more data to train on. 
4. Iterated over hyperparameters again, this time with color data. Managed to find a solution that was barely over 93% target. 
5. When doing the latter part of the lab I realized that when I calculated softmax values I got either ones or zeros. This is because my logits were huge. I hypothesized that these saturated values were crippling my ability to train. So I went back and divided each output by the number of weights. This helped to control my outputs, and I saw a jump of around 1.5% from making this change. 
6. While writing up my architecture in this report, I realized that I forgot to add activation functions on my fully connected layers. This again leaves performance on the table, my three fully connected layers could really be collapsed into one. So I added in relu activation functions. 
7. Adding in the activation functions caused me to take a hit. So I iterated over hyperparameters again. I came up just shy of the target. 
8. I realized that my testing fit was much better than my validation fit, so I decided to add some regularization in the form of two dropout layers after my fully connected layers. This helped to increase my accuracy to the same levels I got before adding in the activation functions. 
9. Went through another round of hyperparameter iteration, improving my validation accuracy to current levels. 

How I iterated over hyperameters:

1. I parameterized the sizes of each linear layer to easily iterate over them. In addition I iterated over hyperparameters. The list of things I iterated over were...

	* learning rate
	* batch size
	* number of epochs
	* depth of first convolutional layer
	* depth of second convolutional layer
	* length of first fully connected layer
	* length of second fully connected layer
	* keep probability for dropouts
2. From the starting values I increased each value by a factor of 2 and decreased it by a factor of two while keeping all other values constant. I trained the network with these adjusted values. At first I stopped training when accuracy stopped improving. Then I realized that there is still interesting information after accuracy levels out, so I switched to training for a fixed number of epochs. 
3. Once all the training was completed. I plotted accuracy vs. time to see the range of accuracies I found and how much time it takes to achieve that accuracy.
 
![accuracyVsTime]

4. Also created a plot of validation accuracy vs epoch to see how the curves behaved
 
![accuracyVsEpoch]

5. Chose the best parameter for each, and used this as my new set of starting values. 
6. Repeat the process as necessary


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose the 5 most commonly mistaken signs (at the time I chose them). 
Here are five German traffic signs that I found on the web:

![webSign0]![webSign1]![webSign2]![webSign3]![webSign4]


The first image skewed since it was taken from an angle. 
The second image might be difficult to classify because of the camoflauge background
The third image is taken straight on, but is rotated. 
The fourth image is both rotated and at an angle. 
The fifth is a pretty straight forward sign. 

I added borders to the images as necessary since the images in the training set have a decent boarder around each sign. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|--------------------|---------------------------------------------| 
| 8) 120km/h     		| 8) 120km/h     								| 
| 2) 50km/h  			| 2) 50km/h										|
| 22) Bumpy road		| 29) Bicycles Crossing							|
| 41) End no passing	| 32) End of all speed and passing limits		|
| 33) Turn right ahead	| 33)Turn right ahead 							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is significantly worse than that of the test set since the images were chosen to be the most commonly mistaken and to push the boundaries of the network. However, the correct answer is always in the top two guesses. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Step 3 of the Ipython notebook.

### Image1: 120km/h
The neural network chose the correct class and was very sure about it. All the top suggestions are other speed limit signs. It handles the skew fine. 

| Probability         	|     Prediction	        					| 
|---------------------|---------------------------------------------| 
| 99.97         			| **8) 120km/h**    									| 
| 0.03     				| 4) 70km/h 										|
| 0					| 7) 100km/h											|
| 0	      			| 5) 80km/h					 				|
| 0				    | 0) 20km/h      							|

### Image2: 50km/h 
The neural network chose the correct class and was very sure about it. All the top suggestions are other speed limit signs. 

| Probability         	|     Prediction	        					| 
|---------------------|---------------------------------------------| 
| 100        			| **2) 50km/h**    									| 
| 0    				| 5) 80km/h 										|
| 0					| 1) 30km/h											|
| 0	      			| 3) 60km/h					 				|
| 0				    | 7) 100km/h      							|

### Image3: Bumpy road
The network is unsure about the class and chooses incorrectly. The top selection is another red triangle with some black stuff in the middle. Besides the 60km/h sign, all the other top suggestions are red boardered triangles.

| Probability         	|     Prediction	        					| 
|---------------------|---------------------------------------------| 
| 54.9        			| 29) Pedestrians   									| 
| 14.3     				| **22) Bumpy road**  										|
| 9.5					| 3) 60km/h	 											|
| 7.3	      			| 25) Road work 					 				|
| 6.2				    | 13) Yield      							|

### Image4: End no passing
The network is unsure about the class and chooses incorrectly. The correct choice is very close in probability to the first choice. And at the image's angle and pixelation, it is very easy for even a human to misclassify this one. Besides 12, all the remaing top ranked classes are circles with a slash through them. 

| Probability         	|     Prediction	        					| 
|---------------------|---------------------------------------------| 
| 34.7         			| 32) End of all speed and passing limits   									| 
| 32.7     				| **41) End no passing**  										|
| 15.7					| 12) Priority road											|
| 7.6	      			| 6) End of speed limit (80km/h)				 				|
| 4.6				    | 42) End no passing over 3.5 metric tons      							|

### Image5: Turn right ahead
The neural network chose the correct class and was very sure about it. All the top suggestions are blue circles

| Probability         	|     Prediction	        					| 
|---------------------|---------------------------------------------| 
| 100        			| **33) Turn right ahead**    									| 
| 0    				| 40) Roundabout mandatory 										|
| 0					| 39) Keep left											|
| 0	      			| 35) Ahead only					 				|
| 0				    | 37) Go straight or left      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I didn't do this.

---

## Reflections

There is so much more that I want to do in this lab, but unfortunately I won't have the time to. I've already hit the target multiple times, then went back to tweak things to make things better. Unfortunately I have to submit this lab at some point. 

But here's a list of things I would like to do if I end up having the time to come back to this lab and try stuff out. 

* Augment the data set: help it to deal with rotations, which my network struggled on with the images downloaded from the internet: 
* Batch normalization: heard about this in one of the Stanford lectures recommended earlier in the course. This would help with previous issues where my logits were huge, causing me to get all zero or one probabilities. 
* More instrumentation in the model: I'd like to take a look at how the weights and outputs change from layer to layer
* Playing around with tensorflow: It looks like there are a bunch of visualization and instrumentation features of tensorflow that I haven't begun to touch yet. 
* The optional section: I'm curious what this model ended up detecting. 
