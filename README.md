# Logistic Regression

Implementation of Logistic Regression using Gradient Descent based on deeplearning.ai's course Neural Networks and Deep Learning (https://www.coursera.org/learn/neural-networks-deep-learning).

The `logistic_regression.py` file contains the implementation (in a scikit-learn-like interface), while the `main.py` file uses it to train a classifier using the data of Kaggle's `Dogs vs Cats` competition (https://www.kaggle.com/c/dogs-vs-cats/data), which contains 25.000 images of cats and dogs. To get image features, i used ResNet50 with Keras, of which i discarded the last layer and kept the last flatten layer, getting a 2048-dimensional vector per image.

Dataset split:
1. 70% Training set (of which 10% is used for validation - the Validation set)
2. 30% Test set

The model is trained using a learning rate of 0.01 and 1000 iterations. Below you can see the training and validation losses and accuracies (iterations in hundreds):

![Alt text](loss_accuracy_plot.png?raw=true "Loss and Accuracy curve")

Accuracy in test set: 0.989333

And here are some predictions (class and probability) for 25 random test images:
![Alt text](image_plot.png?raw=true "Class and Probability predictions for random images")
