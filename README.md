# Simple_Neural_Network
##Fashion MNIST Classification with TensorFlow and Keras
This project involved developing and evaluating a neural network model to classify images from the Fashion MNIST dataset using TensorFlow and Keras. The dataset consists of grayscale images of ten different categories of clothing and accessories.
### Table of Contents
-Overview
- Installation
- Dataset
 Model Architecture
  Training the Model
  Evaluation
  Results
  Conclusion
  Acknowledgements
## Overview
The goal of this project was to create a neural network capable of accurately classifying images into one of the ten categories provided in the Fashion MNIST dataset. The project involved the following steps:

 Loading and inspecting the dataset.
Normalizing the image data.
Defining the neural network model.
Training the model with validation and early stopping.
Evaluating the model's performance.
Visualizing training progress and results.

### Installation
To run this project, I needed to have Python installed along with the following libraries:

TensorFlow
Keras
NumPy
Pandas
Matplotlib
Scikit-learn
I installed the required libraries using pip:
## Dataset
The Fashion MNIST dataset is a collection of 70,000 grayscale images of 28x28 pixels, categorized into ten different classes. The dataset is split into 60,000 training images and 10,000 test images. Each image corresponds to a specific category of clothing or accessory.

## Model Architecture
The neural network model was defined using TensorFlow and Keras. It consisted of:

A flatten input layer to convert 2D images into a 1D vector.
A dense hidden layer with 128 neurons and ReLU activation.
An output layer with 10 neurons, each representing a class in the dataset.
The model was compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function, with accuracy as the evaluation metric.

## Training the Model
I trained the model using the normalized training data for ten epochs. Early stopping and model checkpoint callbacks were used to optimize training and prevent overfitting. The training process included validation on 20% of the training data.
Evaluation
I evaluated the model's performance on the test dataset, and recorded key metrics such as accuracy and loss. The results were visualized to understand the model's learning progress.

## Results
The model achieved satisfactory accuracy on the test dataset. I plotted training and validation accuracy and loss to visualize the learning process. Additionally, I generated a confusion matrix to assess the model's classification performance across different categories.

## Conclusion
This project successfully demonstrated the application of a neural network for image classification within the fashion domain. The model showed promising results, achieving high accuracy in predicting fashion categories. Future work could focus on further optimizing the model architecture, expanding the dataset, and exploring advanced techniques such as convolutional neural networks (CNNs) to improve classification performance.

## Acknowledgements
This project utilized the Fashion MNIST dataset provided by Zalando Research. The implementation was made possible using TensorFlow and Keras libraries.
