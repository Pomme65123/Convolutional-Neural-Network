# MNIST Convolutional Neural Network

A from-scratch C++ implementation of a Convolutional Neural Network (CNN) for MNIST digit classification. This project demonstrates the fundamental building blocks of a CNN without relying on deep learning frameworks.

## Features

- Pure C++ implementation with no ML frameworks
- Custom 2D tensor operations
- Convolutional layers with 3x3 kernels
- ReLU activation functions
- Max pooling layers
- Fully connected layers
- Softmax output layer
- Backpropagation and gradient descent
- Per-class accuracy metrics
- Sample image output generation

## Architecture

The network consists of the following layers:

1. **Input**: 28x28 grayscale images
2. **Convolutional Layer**: 3x3 kernel
3. **ReLU Activation**
4. **Max Pooling**: 2x2 stride
5. **Flatten Layer**
6. **Fully Connected Layer**: 169 -> 10 neurons
7. **Softmax**: 10-class probability distribution

## Prerequisites

- C++ compiler with C++11 support or higher (g++, clang++)
- Make (optional, for build automation)

## Dependencies

### stb_image_write.h

This project uses the `stb_image_write.h` header-only library for saving output images. This file should **not** be committed to the repository (see `.gitignore`).

**Download Instructions:**

```bash
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

Or manually download from: https://github.com/nothings/stb/blob/master/stb_image_write.h

Place the downloaded file in the project root directory.

## Dataset Setup

The project includes a Python script to automatically download and prepare the MNIST dataset in the required binary format.

### Automatic Setup

1. Install Python dependencies:

```bash
pip install torch torchvision numpy
```

2. Run the dataset preparation script:

```bash
python3 MNISTdata.py
```

This script will:
- Download the MNIST dataset using torchvision
- Convert images to binary format (normalized to [0, 1])
- Save files in the `data/` directory:
  - `X_train.bin` - Training images (60,000 images, 28x28 pixels)
  - `y_train.bin` - Training labels
  - `X_test.bin` - Test images (10,000 images)
  - `y_test.bin` - Test labels
  - `metadata.txt` - Dataset information
- Clean up temporary download files

### Manual Setup

If you prefer to prepare the data manually, ensure the binary files are in the `data/` directory with each image stored as 784 floats (28x28), normalized to [0, 1].

## Building and Running

### Compilation

```bash
g++ -o mnist_cnn main.cpp -std=c++11
```

For debugging:

```bash
g++ -o mnist_cnn main.cpp -std=c++11 -g
```

### Execution

```bash
./mnist_cnn
```

## Output

The program will:

1. Load training and test datasets
2. Initialize random kernel weights and FC layer weights using Kaiming initialization
3. Train on all training images for one epoch
4. Evaluate performance on the test set
5. Test random images from each digit class (0-9)
6. Save sample predictions as PNG images in `sample_images/`

## Project Structure

```
.
├── main.cpp              # Main CNN implementation
├── MNISTdata.py          # Python script to download and prepare MNIST data
├── stb_image_write.h     # Image writing library
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── data/                 # MNIST dataset directory
│   ├── X_train.bin       # Training images
│   ├── y_train.bin       # Training labels
│   ├── X_test.bin        # Test images
│   ├── y_test.bin        # Test labels
│   └── metadata.txt      # Dataset information
└── sample_images/        # Generated sample predictions
```

## Implementation Details

### Layers

- **Convolution**: Standard 2D convolution with stride 1, no padding
- **ReLU**: Element-wise `max(0, x)` activation
- **Max Pooling**: 2x2 pooling with stride 2
- **Fully Connected**: Matrix multiplication + bias
- **Softmax**: Numerically stable implementation with max subtraction

### Training

- **Loss Function**: Cross-entropy loss
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.01
- **Batch Size**: 1 (online learning)
- **Weight Initialization**: Kaiming/He initialization for FC layer, uniform for conv kernel

### Backpropagation

The implementation includes custom backward passes for:
- ReLU activation
- Max pooling (gradient routing)
- Fully connected layer
- Softmax + cross-entropy (combined gradient)

## Performance

Typical accuracy after one epoch: ~85-90% on MNIST test set.
