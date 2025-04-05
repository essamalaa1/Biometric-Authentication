# Siamese Network for Biometric Authentication
This project implements a Siamese neural network for biometric authentication using facial images. The system is designed to verify a user's identity by comparing image pairs and determining whether they belong to the same individual.

## Overview
The core idea behind this project is to leverage a Siamese network architecture that learns an embedding space where similar images (i.e., images of the same person) are close together and dissimilar images are far apart. This allows the system to perform authentication by comparing the similarity score between a reference image and a user-provided input image.

## Key Features
Pair Generation: Automatic creation of positive (same person) and negative (different person) image pairs.

Balanced Negative Sampling: Ensures a balanced training dataset by randomly sampling negative pairs.

Image Preprocessing: Images are loaded, resized, normalized, and converted to arrays suitable for model training.

Custom Network Architecture: Uses a convolutional neural network (CNN) backbone to extract image features.

Distance Metric: Utilizes Euclidean distance to measure similarity between image embeddings.

Model Training & Evaluation: Includes training routines, validation metrics, and visualization of loss, accuracy, confusion matrix, and classification report.

User Authentication: Provides a function to authenticate users based on a similarity threshold.

## Technologies and Frameworks
Python: The primary programming language used.

TensorFlow & Keras: Used for building and training the deep learning model, including custom layers and callbacks.

NumPy: For numerical operations and array manipulations.

OpenCV / PIL: For image processing (using Keras preprocessing functions such as load_img and img_to_array).

Matplotlib & Seaborn: For visualizing training metrics, confusion matrices, and sample image pairs.

scikit-learn: Provides utilities for generating classification reports and computing the confusion matrix.

## Project Structure
### Data Preparation:

Generates image paths and creates pairs (both positive and negative) for training and testing.

Balances the dataset with negative sampling to ensure robust training.

### Model Architecture:

Defines a base CNN network for feature extraction.

Constructs the Siamese network that combines twin networks using a Lambda layer for Euclidean distance computation.

### Training & Evaluation:

Compiles the model with a binary cross-entropy loss function and an Adam optimizer.

Trains the model while monitoring performance with a checkpoint callback.

Evaluates the model using accuracy, confusion matrix, and classification reports.

### User Authentication:

Implements a function to authenticate users by comparing a reference image with an input image.

Displays the comparison results with images and similarity scores.
