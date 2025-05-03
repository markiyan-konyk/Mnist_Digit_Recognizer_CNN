# 🧠 MNIST Digit Classifier with PyTorch 🚀

Welcome to a fast-paced, GPU-accelerated tour into digit recognition using deep learning! This project is a **Convolutional Neural Network (CNN)** trained on the legendary MNIST dataset, designed to recognize handwritten digits with high accuracy — and do it live!

> 🔥 Built with PyTorch  
> 🎯 Trained in seconds  
> 🤖 Predicts in real-time  
> 🖼️ Visualizes predictions as they happen  

---

## 🛠️ What’s Inside?

- **Model**: A 2-layer CNN with ReLU activations and max pooling.
- **Training**: 3 epochs with Adam optimizer and cross-entropy loss.
- **Evaluation**: Displays the test set accuracy after training.
- **Prediction Demo**: Randomly selects test images and shows live predictions with matplotlib.
- **Device-Aware**: Automatically uses GPU if available (`cuda` support)!

---

## 🧩 Architecture
Input (1x28x28 grayscale)
├── Conv2d(1, 32, kernel=5) + ReLU + MaxPool2d(2)
├── Conv2d(32, 64, kernel=5) + ReLU + MaxPool2d(2)
├── Flatten
├── Linear(7764 → 256) + ReLU
└── Linear(256 → 10) → Prediction

## 🚀 How to Run

```bash
just copy the file

# Install dependencies
pip install torch torchvision matplotlib tqdm opencv

# Run the script
python MNIST_Webcam.py
