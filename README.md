# Generative-AI-model
# 🧠 generative-AI- | Deep Learning on MNIST

Welcome to the generative-AI- repository! This project showcases core deep learning models — from basic classifiers to generative models — implemented on the MNIST handwritten digit dataset.

---

## 📌 What's Inside

This project explores:

- 🔢 *Basic Neural Network (MLP)*
- 🧠 *Convolutional Neural Network (CNN)*
- 🌀 *Variational Autoencoder (VAE)*
- 🎨 *Generative Adversarial Network (GAN)*

---

## 🗂 Project Structure

| File | Description |
|------|-------------|
| basic-neural-network.ipynb | Implements a simple fully connected network |
| CNN_ON_MNIST_DATASET.ipynb | Builds a CNN classifier for digit recognition |
| variational_encoder_on_mnist_dataset.ipynb | Trains a VAE for digit reconstruction |
| gan-mnist-keras.ipynb | Trains a GAN to generate new MNIST digits |

---

## 🧠 Model Details

### 🔸 Basic Neural Network
- Input: 784 (28x28 flattened pixels)
- Hidden layers with ReLU activation
- Output: 10 softmax classes
- Optimizer: Adam
- Loss: Categorical Crossentropy

### 🔸 Convolutional Neural Network (CNN)
- 2D Convolution + MaxPooling layers
- Fully connected layers for classification
- High test accuracy (~99%)
- Trained using Keras/TensorFlow

### 🔸 Variational Autoencoder (VAE)
- Encoder compresses to latent space (mean, variance)
- Decoder reconstructs digits from latent vectors
- Latent sampling enables image generation
- Uses reparameterization trick

### 🔸 Generative Adversarial Network (GAN)
- Generator learns to create realistic digits from noise
- Discriminator learns to distinguish real vs fake
- Trained using adversarial loss
- Generator improves until it fools the Discriminator
- Implemented using Keras

---
