***My first Convolutional neural network (CNN)***

Here is my attempt as of 05/01 to understand how CNNs work.

In order to grasp how to create and train a CNN, I will use the [KMNIST dataset](https://github.com/rois-codh/kmnist)
, and define my own models, iteratively trying until the CNN emerges with a good enough
performance 
(I guess a good threshold might be a 95% accurate CNN? It might be a whole trip of its own)


Below will also be the general idea for my coding journey.

**Task definition**
Our goal is to create a CNN which takes in KMNIST images and classifies the label of 
those images with some accuracy.

Supervised learning on images.
- Dataset: KMNIST dataset, 10 classes of 7000 images each.
- Input: 3x28x28 image.
- Output: probability that the image is in one of the 10 categories.

**Rough flow of work**
- Divide data into training, testing and validation sets (60 vs 20 vs 20).
- Define a CNN model. This should be the most learning-ful part.
- Training:
1) Initialize the model
2) Feed images as batch into the network to calculate predictions
3) Obtain the loss by comparing with the actual labels
4) Model learns via parameter adjustment with loss
5) The capability to generalize is monitored via loss calculated upon the testing set
6) Process repeated until training loss goes down and testing loss stays low
7) Evaluate the ultimate capability through loss onto validation set
To be iterated over different hyperparameters (epoch, learning rate, etc.)

Credits:
The dataset and work related is attributed as below:
"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
