# This project has 2 parts



# Part-1. Transfer learning to create models for gesture detection
## Dataset can be found on

https://www.kaggle.com/grassknoted/asl-alphabet
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.
![](https://github.com/aakashs11/SLI/blob/master/label%20count.png)

## VGG 16 
This model achieved an accuracy of **91.1%**

###Trained on 30000 samples, validated on 17400 samples


![](https://github.com/aakashs11/SLI/blob/master/Images/VGG16/accuracy.png)

![](https://github.com/aakashs11/SLI/blob/master/Images/VGG16/loss.png)

![](https://github.com/aakashs11/SLI/blob/master/Images/VGG16/cnfmatrix.png)

## VGG 19
Used just as a good classification architecture for many other datasets and as the authors made the models available to the public they can be used as is or with modification for other similar tasks also.
Transfer learning : can be used for facial recognition tasks also.

### Trained on 30000 samples, validated on 17400 samples

Model was trained for 10 epochs after which I unfroze the last 2 convolutional layers(to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task)\
As you are training a much larger model and want to readapt the pretrained weights, it is important to use a lower learning rate at this stage.
Otherwise, your model could overfit very quickly.\
There was an increase of about 5% in validation accuracy. Some overfitting is observed since val loss is a bit higher than training loss.  
<br />
**Final accuracy score of 93%**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| A            | 0.90      | 0.94   | 0.92     | 615     |
| B            | 0.90      | 0.95   | 0.92     | 590     |
| C            | 0.99      | 0.99   | 0.99     | 581     |
| D            | 0.97      | 0.99   | 0.98     | 560     |
| E            | 0.91      | 0.88   | 0.89     | 572     |
| F            | 0.95      | 0.97   | 0.96     | 602     |
| G            | 0.97      | 0.94   | 0.96     | 622     |
| H            | 0.96      | 0.97   | 0.97     | 614     |
| I            | 0.93      | 0.92   | 0.92     | 598     |
| J            | 0.97      | 0.95   | 0.96     | 616     |
| K            | 0.90      | 0.92   | 0.91     | 610     |
| L            | 0.96      | 0.99   | 0.98     | 623     |
| M            | 0.86      | 0.91   | 0.88     | 538     |
| N            | 0.93      | 0.91   | 0.92     | 604     |
| O            | 0.97      | 0.95   | 0.96     | 607     |
| P            | 0.95      | 0.98   | 0.96     | 630     |
| Q            | 0.99      | 0.96   | 0.97     | 626     |
| R            | 0.87      | 0.82   | 0.85     | 593     |
| S            | 0.83      | 0.88   | 0.85     | 600     |
| T            | 0.96      | 0.91   | 0.93     | 586     |
| U            | 0.82      | 0.84   | 0.83     | 581     |
| V            | 0.85      | 0.87   | 0.86     | 594     |
| W            | 0.90      | 0.90   | 0.90     | 562     |
| X            | 0.93      | 0.81   | 0.86     | 651     |
| Y            | 0.89      | 0.96   | 0.92     | 576     |
| Z            | 0.98      | 0.96   | 0.97     | 634     |
| del          | 0.96      | 0.96   | 0.96     | 604     |
| nothing      | 1.00      | 1.00   | 1.00     | 606     |
| space        | 0.97      | 0.95   | 0.96     | 605     |
| accuracy     |           |        | 0.93     | 17400   |
| macro avg    | 0.93      | 0.93   | 0.93     | 17400   |
| weighted avg | 0.93      | 0.93   | 0.93     | 17400   |
|              |           |        |          |         |

![](https://github.com/aakashs11/SLI/blob/master/Images/VGG19/cnf.png)

Performance can be improved by increasing the size of the images, and training for more epochs.

## Xception
Xception  by Google, stands for Extreme version of Inception. With a modified depthwise separable convolution, it is even better than Inception-v3.
### Trained on 30000 samples, validated on 17400 samples
Model was trained fro 5 epochs using a degrading gradient and most of the layers were unfrozen, since the image size was only 32x32. The model was trained again for 5 epochs after changing the learning rate to 1/100th of the initial.

**Final accuracy score of 97.57%**
![](https://github.com/aakashs11/SLI/blob/master/Images/Xception/loss%20and%20accuracy.png)

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| A            | 0.99      | 0.95   | 0.97     | 608     |
| B            | 0.97      | 0.96   | 0.97     | 632     |
| C            | 1.00      | 0.98   | 0.99     | 628     |
| D            | 0.98      | 1.00   | 0.99     | 619     |
| E            | 0.97      | 0.96   | 0.96     | 563     |
| F            | 0.98      | 1.00   | 0.99     | 586     |
| G            | 1.00      | 0.99   | 0.99     | 580     |
| H            | 0.98      | 0.96   | 0.97     | 599     |
| I            | 0.98      | 0.99   | 0.99     | 606     |
| J            | 0.99      | 1.00   | 0.99     | 615     |
| K            | 0.97      | 0.97   | 0.97     | 595     |
| L            | 0.99      | 1.00   | 0.99     | 607     |
| M            | 0.98      | 0.99   | 0.98     | 599     |
| N            | 1.00      | 0.99   | 1.00     | 620     |
| O            | 0.99      | 0.99   | 0.99     | 578     |
| P            | 1.00      | 0.97   | 0.98     | 604     |
| Q            | 0.97      | 1.00   | 0.99     | 586     |
| R            | 0.95      | 0.95   | 0.95     | 593     |
| S            | 0.97      | 0.99   | 0.98     | 558     |
| T            | 0.98      | 0.98   | 0.98     | 637     |
| U            | 0.96      | 0.92   | 0.93     | 589     |
| V            | 0.91      | 0.96   | 0.93     | 627     |
| W            | 0.95      | 0.95   | 0.95     | 615     |
| X            | 0.99      | 0.90   | 0.94     | 621     |
| Y            | 0.91      | 0.99   | 0.95     | 536     |
| Z            | 0.99      | 1.00   | 0.99     | 592     |
| del          | 0.98      | 0.99   | 0.98     | 605     |
| nothing      | 1.00      | 1.00   | 1.00     | 593     |
| space        | 0.99      | 1.00   | 1.00     | 609     |
| accuracy     |           |        | 0.98     | 17400   |
| macro avg    | 0.98      | 0.98   | 0.98     | 17400   |
| weighted avg | 0.98      | 0.98   | 0.98     | 17400   |
|              |           |        |          |         |
![](https://github.com/aakashs11/SLI/blob/master/Images/Xception/cnf.png)

Performance can be greatly improved to about 99.95% by trainnig for 20 epochs.
# 2. Gesture Recognition using OpenCV and Deep CNN

## About the Project

Here we are extracting the images of the hand in Grayscale using absolute background subtraction in OpenCV. For that you have to select the histogram which suits the best. You might need to make some minute changes in the code for it to work on your machine.


## Dataset can be found on

https://drive.google.com/open?id=1tQuO_-lGpt9Qwxjp3F6rqF_k7P3_8Eef

Collection of 7 custom gestures, which can be further extended to add Indian Sign Language Vocabulary.

## File Description

[Add_Gesture.py](https://github.com/aakashsingh11/Sign-Language-Interpretation/blob/master/Add_Gesture.py): Run this file to add gestures. It can be used to add vocabulary in any sign language. Go into the file and change the name of the directory and make other appropriate changes.

[ResizeImages.py](https://github.com/aakashsingh11/Sign-Language-Interpretation/blob/master/ResizeImages.py): Run this file after Add_Gesture.py in order to resize the images so that it can be fed into the Convolution Neural Network designed using tensorflow. The network accepts 89 x 100 dimensional image.

[Train.py](https://github.com/aakashsingh11/Sign-Language-Interpretation/blob/master/Train.py): Run this file  to retrain the model using your custom dataset.

[Predictor.py](https://github.com/aakashsingh11/Sign-Language-Interpretation/blob/master/Predictor.py): Running this file opens up your webcam and takes continuous frames of your hand image and then predicts the class of your hand gesture in realtime.

## Architecture

### Background Ellimination 

After taking the average of 30 frames you can introduce your hand into the frame and save the histogram. This will effectively subtract the background and show only the black and white pixels.

Refer: [Gogul09](https://github.com/Gogul09) 


### The Deep Convolution Neural Network

The network contains **7** hidden convolution layers with **Relu** as the activation function and **1** Fully connected layer.

The network is trained across **50** iterations with a batch size of **64**.

I kind of saw that 50 iterations kind of trains the model well and there is no increase in validation accuracy along the lines so that should be enough.

The model achieves an accuracy of **99.2%** on the validation dataset.

The ratio of training set to validation set is **1000 : 100**.

## How to run the RealTime prediction

Run the [Predictor.py](https://github.com/aakashsingh11/Sign-Language-Interpretation/blob/master/Predictor.py) and you will see a window named **Video Feed** appear on screen. Wait for a while until a window named **Thresholded** appears.

The next step involves pressing **"s"** on your keyboard in order to start the real-time prediction.

Bring your hand in the **Green Box** drawn inside **Video Feed** window in order to see the predictions.
Look in demo for some visual clarity.

## Demo of how things look on the go

Well now it's time for some demo.

![](https://github.com/aakashs11/SLI/blob/master/Gesture%20Predictor.gif)

## Requirements

* Python3
* Tensorflow
* TfLearn
* Opencv (cv2) for python3
* Numpy
* Pillow (PIL)
* Imutils
