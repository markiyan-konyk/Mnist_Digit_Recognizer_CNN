# MNIST Digit Classifier with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model achieves high accuracy after just a few training epochs, although some further optimization would be needed, and includes a simple interactive demo to visualize predictions.

## Instructions

- Let the program train the model, progress bars wil appear, and the accuracy will be shown.
- Two windows will appear, a big coloured one with a green square in which to centre the drawn digit. Another smaller window, will show the B&W image that the CNN sees.
- Press the 'q' key to take a screenshot of the B&W image.
- Press the 'e' key to exit the program.

## Features

- Two-layer CNN with ReLU activations and max pooling
- Trained on the MNIST dataset using the Adam optimizer
- Achieves ~98% test accuracy after 3 epochs, although 5-7 are recommended.
- Interactive visualization of random test predictions
- Automatically utilizes GPU if available

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm
- opencv

Install dependencies with:

```bash
pip install torch torchvision matplotlib tqdm opencv
