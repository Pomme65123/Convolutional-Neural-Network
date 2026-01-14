#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <ctime>

// Used for weight initialization, otherwise every recompile will result in the same numbers.
std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));

/*
    Using: https://github.com/nothings/stb/blob/master/stb_image_write.h
    to convert .bin files into .png files.
*/
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Reading float/int values from .bin files
std::vector<float> readFloatFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open " + path);
    in.seekg(0, std::ios::end);
    std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<float> buf(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;
}

std::vector<int> readIntFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open " + path);
    in.seekg(0, std::ios::end);
    std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<int> buf(bytes / sizeof(int));
    in.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;
}


// MNIST data structure uses 2D images with 1 channel
// For larger data can change from int -> size_t
struct Tensor2D {
    int rows, cols;
    std::vector<float> data;

    Tensor2D(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    inline float& operator()(int r, int c) { return data[r * cols + c]; }
    inline const float& operator()(int r, int c) const { return data[r * cols + c]; }

    int Rows() const { return rows; }
    int Cols() const { return cols; }
    int Size() const { return static_cast<int>(data.size()); }  // No need for size_t

    friend std::ostream& operator<<(std::ostream& os, const Tensor2D& matrix);
};

// Friend function to make debugging easier
std::ostream& operator<<(std::ostream& os, const Tensor2D& matrix) {
    for (int row{}; row < matrix.Rows(); row++) {
        for (int col{}; col < matrix.Cols(); col++) {
            os << matrix(row, col);
            if (col + 1 < matrix.Cols()) os << " ";
        }
        os << '\n';
    }
    return os;
}

struct FullyConnectedGrad {
    std::vector<std::vector<float>> dWeights;
    std::vector<float> dBias;
    std::vector<float> dInput;
};

// 2D Convolutional Layer
// https://en.wikipedia.org/wiki/Convolutional_layer
Tensor2D convolutionLayer (const Tensor2D& matrix, const Tensor2D& kernel) {
    // int rowsMatrix = matrix.Rows();
    // int colsMatrix = matrix.Cols();
    // int rowskernel = kernel.Rows();
    // int colskernel = kernel.Cols();

    // int rowsResult = rowsMatrix - rowskernel + 1;
    // int colsResult = colsMatrix - colskernel + 1;

    int rowsResult = matrix.Rows() - kernel.Rows() + 1;
    int colsResult = matrix.Cols() - kernel.Cols() + 1;
    Tensor2D result(rowsResult, colsResult);

    for (int rows{}; rows < rowsResult; rows++) {
        for (int cols{}; cols < colsResult; cols++) {
            float sum = 0.0f;
            for (int m{}; m < kernel.Rows(); m++) {
                for (int n{}; n < kernel.Cols(); n++) {
                    sum += matrix(rows + m, cols + n) * kernel(m,n);
                }
            }
            result(rows,cols) = sum;
        }
    }

    return result;
}

// ReLu Layer
// https://en.wikipedia.org/wiki/Rectified_linear_unit
Tensor2D reluLayer (const Tensor2D& matrix) {
    // int rowsMatrix = matrix.Rows();
    // int colsMatrix = matrix.Cols();

    // Tensor2D result(rowsMatrix, colsMatrix);
    
    // const int size = matrix.Size();

    Tensor2D result(matrix.Rows(), matrix.Cols());

    for (int i{}; i < matrix.Size(); i++) {
        float val = matrix.data[i];
        result.data[i]= (val > 0.0f) ? val : 0.0f;
    }

    return result;
}

// ReLu Layer for backpropagation
// Basically if val > 0, assign to the previous data
Tensor2D backwardReluLayer(const Tensor2D& matrix, const Tensor2D& dOut) {
    // int rows = matrix.Rows();
    // int cols = matrix.Cols();
    // Tensor2D dX(rows, cols);

    // const int size = matrix.Size();

    Tensor2D dX(matrix.Rows(),matrix.Cols());

    for (int i{}; i < matrix.Size(); i++) {
        float val = matrix.data[i];
        dX.data[i] = (val > 0.0f) ? dOut.data[i] : 0.0f;
    }
    return dX;
}

// Pooling Layer
// https://en.wikipedia.org/wiki/Pooling_layer
Tensor2D maxPoolingLayer (const Tensor2D& matrix, int stride = 2) {
    int rows = matrix.Rows();
    int cols = matrix.Cols();

    int outputHeight = rows / stride;
    int outputWidth = cols / stride;

    Tensor2D result(outputHeight,outputWidth);

    for (int row{}; row < outputHeight; row++) {
        for (int col{}; col < outputWidth; col++) {
            int h = row * stride;
            int w = col * stride;

            float m = matrix(h, w);
            m = std::max(m, matrix(h, w + 1));
            m = std::max(m, matrix(h + 1, w));
            m = std::max(m, matrix(h + 1, w + 1));
            result(row,col) = m; 
        }
    }
    
    return result;
}

// Pooling Layer for backpropagation
// https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec/
Tensor2D backwardMaxPooling(const Tensor2D& inMatrix,
                            const Tensor2D& dOut,
                            int stride = 2) {

    int rows = inMatrix.Rows();
    int cols = inMatrix.Cols();
    int outHeight = dOut.Rows();
    int outWidth = dOut.Cols();

    Tensor2D dX(rows, cols);

    for (int row{}; row < outHeight; row++) {
        for (int col{}; col < outWidth; col++) {
            int h = row * stride;
            int w = col * stride;

            float m = inMatrix(h, w);
            int max_r = h;
            int max_c = w;

            if (inMatrix(h, w + 1) > m) { m = inMatrix(h, w + 1); max_r = h; max_c = w + 1; }
            if (inMatrix(h + 1, w) > m) { m = inMatrix(h + 1, w); max_r = h + 1; max_c = w; }
            if (inMatrix(h + 1, w + 1) > m) { m = inMatrix(h + 1, w + 1); max_r = h + 1; max_c = w + 1; }

            dX(max_r, max_c) += dOut(row, col);
        }
    }

    return dX;
}

// Fully Connected Layer
// https://www.geeksforgeeks.org/deep-learning/what-is-fully-connected-layer-in-deep-learning/
std::vector<float> fullyConnectedLayer( const std::vector<float>& inVec,
                                        const std::vector<std::vector<float>>& weight,
                                        const std::vector<float>& bias) {

    std::size_t out_dim = weight.size();
    std::size_t in_dim  = inVec.size();
    std::vector<float> outVec(out_dim);

    for (std::size_t neuron{}; neuron < out_dim; neuron++) {
        float sum = bias[neuron];
        for (std::size_t idx{}; idx < in_dim; idx++) {
            sum += weight[neuron][idx] * inVec[idx];
        }
        outVec[neuron] = sum;
    }
    
    return outVec;
}

/*  Dense Layer Back Propagation
    https://cs231n.stanford.edu/handouts/linear-backprop.pdf
    Works for Batch Size = 1
*/
FullyConnectedGrad fullyConnectedBackward(  const std::vector<float>& matrix,
                                            const std::vector<std::vector<float>>& weight,
                                            const std::vector<float>& dLogits) {

    std::size_t outDim = weight.size();
    std::size_t inDim  = matrix.size();

    FullyConnectedGrad grad;
    grad.dWeights.assign(outDim, std::vector<float>(inDim, 0.0f));
    grad.dBias.resize(outDim);
    grad.dInput.assign(inDim, 0.0f);

    // for (std::size_t i{}; i < outDim; i++) {
    //     float g = dLogits[i];
    //     grad.dBias[i] += g;
    //     for (std::size_t j{}; j < inDim; j++) {
    //         grad.dWeights[i][j] += g * matrix[j];
    //     }
    // }

    // for (std::size_t i{}; i < inDim; i++) {
    //     float sum = 0.0f;
    //     for (std::size_t j{}; j < outDim; j++) {
    //         sum += weight[j][i] * dLogits[j];
    //     }
    //     grad.dInput[i] = sum;
    // }
    
    for (std::size_t i{}; i < outDim; i++) {
        float g = dLogits[i];
        grad.dBias[i] = g;
        for (std::size_t j{}; j < inDim; j++) {
            grad.dWeights[i][j] = g * matrix[j];
            grad.dInput[j] += weight[i][j] * g;
        }
    }

    return grad;
}

// Softmax Layer
// https://en.wikipedia.org/wiki/Softmax_function
std::vector<float> softmaxLayer(const std::vector<float>& inVec) {
    
    std::size_t size = inVec.size();
    std::vector<float> outVec(size);
    float maxVal = *std::max_element(inVec.begin(), inVec.end());

    float sum = 0.0f;
    
    // for (std::size_t i{}; i < size; i++) {
    //     outVec[i] = std::exp(inVec[i] - maxVal);
    //     sum += outVec[i];
    // }

    // for (std::size_t i{}; i < size; i++) {
    //     outVec[i] /= sum;
    // }

    for (std::size_t i{}; i < size; i++) {
        outVec[i] = std::exp(inVec[i] - maxVal);    // Prevents overflow as the largest value will be 0
        sum += outVec[i];
    }

    // Optimization technique because multiplication is faster than division in loops
    float invSum = 1.0f / sum;
    for (std::size_t i{}; i < size; i++) {
        outVec[i] *= invSum;
    }

    return outVec;
}

Tensor2D convBackwardKernel(const Tensor2D& input,
                            const Tensor2D& dOut,
                            int kRows, int kCols) {

    Tensor2D dKernel(kRows, kCols);

    int outRows = dOut.Rows();
    int outCols = dOut.Cols();

    for (int krow{}; krow < kRows; krow++) {
        for (int kcol{}; kcol < kCols; kcol++) {
            float sum = 0.0f;
            for (int row{}; row < outRows; row++) {
                for (int col{}; col < outCols; col++) {
                    sum += input(row + krow, col + kcol) * dOut(row, col);
                }
            }
            dKernel(krow, kcol) = sum;
        }
    }
    
    return dKernel;
}

// CNN Forward Pass
std::vector<float> forward( const Tensor2D& image,
                            const Tensor2D& kernel,
                            const std::vector<std::vector<float>>& weight,
                            const std::vector<float>& bias) {
    
    Tensor2D outConv = convolutionLayer(image, kernel);
    Tensor2D outRelu = reluLayer(outConv);
    Tensor2D outPool = maxPoolingLayer(outRelu);

    const std::vector<float>& flattenedData = outPool.data;
    std::vector<float> outForward = fullyConnectedLayer(flattenedData, weight, bias);
    std::vector<float> outSoftMax = softmaxLayer(outForward);

    return outSoftMax;
}

// Prediction Function
// Goes through a vector of probabilities and returns the highest valued index
int predictClass(const std::vector<float>& probs) {
    std::size_t best = 0;
    for (std::size_t i{}; i < probs.size(); i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

// Reshapes float array into a 2D Tensor
Tensor2D makeImage(const std::vector<float>& arr, int index) {
    const int imgSquare = 28;
    // const int imgSize = imgSquare * imgSquare;

    Tensor2D img(imgSquare, imgSquare);

    const float* base = arr.data() + index * (imgSquare * imgSquare);
    for (int row{}; row < imgSquare; row++) {
        for (int col{}; col < imgSquare; col++) {
            img(row, col) = base[row * imgSquare + col];
        }
    }
    return img;
}

// Save MNIST image as PNG format
void saveMNISTImageAsPNG(   const std::vector<float>& arr, 
                            int index, 
                            int label, 
                            const std::string& filename) {

    const int imgSquare = 28;
    const int imgSize = imgSquare * imgSquare;
    const float* base = arr.data() + index * imgSize;
    
    // 8-bit conversion to grayscale
    std::vector<unsigned char> pixels(imgSize);
    for (int i{}; i < imgSize; i++) {
        pixels[i] = static_cast<unsigned char>(base[i] * 255.0f);
    }
    
    if (stbi_write_png(filename.c_str(), imgSquare, imgSquare, 1, pixels.data(), imgSquare)) {
        std::cout << "Saved image (label = " << label << ") to " << filename << "\n";
    } else {
        std::cerr << "Failed to save " << filename << "\n";
    }
}

/*
    Computes loss given a models probabilities and class
    https://en.wikipedia.org/wiki/Cross-entropy
    https://jmlb.github.io/ml/2017/12/26/Calculate_Gradient_Softmax/

    Can replace eps = 1e-12f with: std::numeric_limits<float>::lowest()
    But need #include <limits> and #include <iomanip>
*/
float crossEntropyLoss(const std::vector<float>& probs, int label) {
    const float eps = 1e-12f;   // Prevents log(0)
    float p = std::max(probs[label], eps);
    return -std::log(p);
}
std::vector<float> softmaxCrossEntropyGrad(const std::vector<float>& probs, int label) {
    std::vector<float> grad = probs;
    grad[label] -= 1.0f;
    return grad;
}

// Applies stochastic gradient descent to weights and biases
// https://en.wikipedia.org/wiki/Stochastic_gradient_descent
void sgdUpdate(std::vector<std::vector<float>>& weight,
               std::vector<float>& bias,
               const FullyConnectedGrad& grad,
               float learningRate) {

    const std::size_t out_dim = weight.size();
    const std::size_t in_dim  = weight[0].size();

    for (std::size_t j{}; j < out_dim; j++) {
        bias[j] -= learningRate * grad.dBias[j];
        for (std::size_t i{}; i < in_dim; i++) {
            weight[j][i] -= learningRate * grad.dWeights[j][i];
        }
    }
}

/*
    Test a random image from the dataset
    Using it to test all ten classes (0~9)
    Saves the image as a PNG using: saveMNISTImageAsPNG()
*/
void testRandomImage(const std::vector<float>& X,
                     const std::vector<int>& Y,
                     const Tensor2D& kernel,
                     const std::vector<std::vector<float>>& weight,
                     const std::vector<float>& bias,
                     int index) {
    
    Tensor2D img = makeImage(X, index);
    int trueLabel = Y[index];
    
    std::vector<float> probs = forward(img, kernel, weight, bias);
    int pred = predictClass(probs);
    
    std::cout << "\n=== Testing Image #" << index << " ==="<< "\n";
    std::cout << "True Label: " << trueLabel << "\n";
    std::cout << "Predicted: " << pred << "\n";
    std::cout << "Confidence: " << (probs[pred] * 100.0f) << "%\n";
    std::cout << "\nAll Class Probabilities:\n";

    for (int i{}; i < 10; i++) {
        std::cout << "  Class " << i << ": " << (probs[i] * 100.0f) << "%\n";
    }
    
    std::string filename = "sample_images/test_image_" + std::to_string(index) + "_label_" + std::to_string(trueLabel) + ".png";
    saveMNISTImageAsPNG(X, index, trueLabel, filename);
}

// Training an epoch.
void trainEpoch(const std::vector<float>& X_train,
                   const std::vector<int>& Y_train,
                   Tensor2D& kernel,
                   std::vector<std::vector<float>>& weight,
                   std::vector<float>& bias,
                   float learningrate) {

    int numTrain = static_cast<int>(X_train.size() / (28 * 28));
    float totalLoss = 0.0f;
    int correct = 0;
    
    for (int idx{}; idx < numTrain; ++idx) {
        Tensor2D img = makeImage(X_train, idx);
        int label = Y_train[idx];

        // I can use forward() here
        // I can use forward() here
        // I can use forward() here
        Tensor2D outConv = convolutionLayer(img, kernel);
        Tensor2D outRelu = reluLayer(outConv);
        Tensor2D outPool = maxPoolingLayer(outRelu);

        const std::vector<float>& flat = outPool.data;
        std::vector<float> logits = fullyConnectedLayer(flat, weight, bias);
        std::vector<float> probs = softmaxLayer(logits);
        // I can use forward() here
        // I can use forward() here
        // I can use forward() here
        
        float loss = crossEntropyLoss(probs, label);
        int pred = predictClass(probs);

        totalLoss += loss;
        if (pred == label) correct++;
        
        std::vector<float> dLogits = softmaxCrossEntropyGrad(probs, label);
        FullyConnectedGrad gradFC = fullyConnectedBackward(flat, weight, dLogits);
        
        Tensor2D dPool(outPool.Rows(), outPool.Cols());

        for (int i{}; i < outPool.Size(); i++) dPool.data[i] = gradFC.dInput[i];
        
        Tensor2D dRelu = backwardMaxPooling(outRelu, dPool);
        Tensor2D dConv = backwardReluLayer(outConv, dRelu);
        Tensor2D dKernel = convBackwardKernel(img, dConv, kernel.Rows(), kernel.Cols());
        
        sgdUpdate(weight, bias, gradFC, learningrate);
        
        for (int row{}; row < kernel.Rows(); row++) {
            for (int col{}; col < kernel.Cols(); col++) {
                kernel(row, col) -= learningrate * dKernel(row, col);
            }
        }
        
        if ((idx + 1) % 100 == 0 || idx == numTrain - 1) {
            float avgLoss = totalLoss / (idx + 1);
            float accuracy = 100.0f * correct / (idx + 1);
            std::cout << "Sample " << (idx + 1) << "/" << numTrain << " | Avg Loss: " << avgLoss << " | Accuracy: " << accuracy << "%\n";
        }
    }
}

// Runs the architecture across the entire test set to compute accuracy per class
void evaluateModel(const std::vector<float>& X_test,
                   const std::vector<int>& Y_test,
                   const Tensor2D& kernel,
                   const std::vector<std::vector<float>>& weight,
                   const std::vector<float>& bias) {
    
    int numTest = static_cast<int>(X_test.size() / (28 * 28));
    int correct = 0;
    std::vector<int> classCorrect(10, 0);
    std::vector<int> classTotal(10, 0);
    
    std::cout << "\n=== Evaluating Model ==="<< "\n";
    
    for (int idx{}; idx < numTest; idx++) {
        Tensor2D img = makeImage(X_test, idx);
        int label = Y_test[idx];
        
        std::vector<float> probs = forward(img, kernel, weight, bias);
        int pred = predictClass(probs);
        
        classTotal[label]++;
        if (pred == label) {
            correct++;
            classCorrect[label]++;
        }
        
        if ((idx + 1) % 1000 == 0) std::cout << "Evaluated " << (idx + 1) << "/" << numTest << "\n";
    }
    
    float overallAccuracy = 100.0f * correct / numTest;

    std::cout << "\n=== Test Results ==="<< "\n";
    std::cout << "Overall Accuracy: " << overallAccuracy << "% (" << correct << "/" << numTest << ")\n\n";
    std::cout << "Per-Class Accuracy:\n";

    for (int idxClass{}; idxClass < 10; idxClass++) {
        if (classTotal[idxClass] == 0) {
            std::cout << "Division by Zero\n";
            continue;
        }
        float acc = 100.0f * classCorrect[idxClass] / classTotal[idxClass];
        std::cout << "  Class " << idxClass << ": " << acc << "% (" << classCorrect[idxClass] << "/" << classTotal[idxClass] << ")\n";
    }
}



int main() {

    std::cout << "=== MNIST CNN in C++ ==="<< "\n\n";
    std::cout << "Loading training data...\n";

    // Loading training data
    auto X_train = readFloatFile("data/X_train.bin");
    auto y_train = readIntFile("data/y_train.bin");
    int numTrain = static_cast<int>(X_train.size() / (28 * 28));

    std::cout << "Train images: " << numTrain << "\n";


    // Loading testing data
    std::cout << "Loading test data...\n";
    auto X_test = readFloatFile("data/X_test.bin");
    auto y_test = readIntFile("data/y_test.bin");
    int numTest = static_cast<int>(X_test.size() / (28 * 28));
    std::cout << "Test images: " << numTest << "\n\n";



    // Creates and initializes kernal using a random uniform distribution
    Tensor2D kernel(3, 3);
    std::uniform_real_distribution<float> uni_kernel(-0.1f, 0.1f);

    for (int row{}; row < kernel.Rows(); row++) {
        for (int col{}; col < kernel.Cols(); col++) {
            kernel(row, col) = uni_kernel(rng);
        }
    }

    // Kaiming Initialization for weights
    // https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
    const std::size_t flatDim = 13 * 13;
    float heInit = std::sqrt(2.0f / flatDim);

    std::uniform_real_distribution<float> uni_fc(-heInit, heInit);
    std::vector<std::vector<float>> weight(10, std::vector<float>(flatDim));

    for (int row{}; row < 10; row++) {
        for (std::size_t col{}; col < flatDim; col++) {
            weight[row][col] = uni_fc(rng);
        }
    }

    std::vector<float> bias(10, 0.0f);


    // BEGIN THE EPOCH
    std::cout << "=== Training ==="<< "\n";
    float learningRate = 0.01f;
    trainEpoch(X_train, y_train, kernel, weight, bias, learningRate);

    evaluateModel(X_test, y_test, kernel, weight, bias);
    
    std::cout << "\n=== Testing Random Images from Each Class ==="<< "\n";
    for (int targetClass = 0; targetClass < 10; targetClass++) {
        for (int i{}; i < numTest; i++) {
            if (y_test[i] == targetClass) {
                testRandomImage(X_test, y_test, kernel, weight, bias, i);
                break;
            }
        }
    }
    
    std::cout << "\n=== Wee! ==="<< "\n";

    return 0;
}