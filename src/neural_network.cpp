#include "nanoml/neural_network.hpp"
#include "nanoml/tensor_utils.hpp"
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace ml {
    NeuralNetwork::NeuralNetwork(const std::vector<std::size_t>& layer_sizes) 
        : layer_sizes_(layer_sizes), 
        num_layers_(layer_sizes.size()),
        weights_(num_layers_ - 1),
        biases_(num_layers_ - 1),
        activations_(num_layers_),
        z_values_(num_layers_ - 1),
        rng_(std::random_device{}())
    {

        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (std::size_t i = 0; i < num_layers_ - 1; ++i) {
            weights_[i] = Matrix(layer_sizes_[i+1], layer_sizes_[i]);
            
            double scale = std::sqrt(2.0 / layer_sizes_[i]);
            
            for (std::size_t r = 0; r < weights_[i].rows(); ++r) {
                for (std::size_t c = 0; c < weights_[i].cols(); ++c) {
                    weights_[i](r, c) = dist(rng_) * scale;
                }
            }
            
            biases_[i] = Vector(layer_sizes_[i+1], 0.0);
        }
    }

    Vector NeuralNetwork::forward(const Matrix& X) const {
        assert(X.cols() == layer_sizes_[0]);
        
        Vector current_activation = X.row(0);
        activations_[0] = current_activation;
        
        for (std::size_t i = 0; i < num_layers_ - 1; ++i) {
            // Compute z = W*a + b
            Vector z = weights_[i] * current_activation + biases_[i];
            z_values_[i] = z;
            
            current_activation = Vector(z.size());
            for (std::size_t j = 0; j < z.size(); ++j) {
                current_activation[j] = sigmoid(z[j]);
            }
            
            activations_[i+1] = current_activation;
        }
        
        return current_activation;
    }

    void NeuralNetwork::backward(const Matrix& X, const Vector& grad, double learning_rate) {
        assert(grad.size() == layer_sizes_.back());

        Vector delta = grad;
        
        for (int i = num_layers_ - 2; i >= 0; --i) {
            for (std::size_t j = 0; j < delta.size(); ++j) {
                delta[j] *= sigmoid_grad(activations_[i+1][j]);
            }

            for (std::size_t j = 0; j < biases_[i].size(); ++j) {
                biases_[i][j] -= learning_rate * delta[j];
            }

            for (std::size_t r = 0; r < weights_[i].rows(); ++r) {
                for (std::size_t c = 0; c < weights_[i].cols(); ++c) {
                    // Compute gradient with regularization terms
                    double gradient = delta[r] * activations_[i][c];
                    
                    // Add L2 regularization gradient: 2 * lambda * w
                    if (l2_lambda_ > 0) {
                        gradient += 2 * l2_lambda_ * weights_[i](r, c);
                    }
                    
                    // Add L1 regularization gradient: lambda * sign(w)
                    if (l1_lambda_ > 0) {
                        if (weights_[i](r, c) > 0) {
                            gradient += l1_lambda_;
                        } else if (weights_[i](r, c) < 0) {
                            gradient -= l1_lambda_;
                        }
                        // If weight is exactly 0, the gradient is 0 (or technically, in [-lambda, lambda])
                    }
                    
                    // Update weights with regularization
                    weights_[i](r, c) -= learning_rate * gradient;
                }
            }

            if (i > 0) {
                Vector next_delta(layer_sizes_[i]);
                for (std::size_t j = 0; j < next_delta.size(); ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < delta.size(); ++k) {
                        sum += delta[k] * weights_[i](k, j);
                    }
                    next_delta[j] = sum;
                }
                delta = next_delta;
            }
        }
    }

    bool NeuralNetwork::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        std::string model_type = get_model_type();
        std::size_t type_size = model_type.size();
        file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
        file.write(model_type.c_str(), type_size);

        file.write(reinterpret_cast<const char*>(&num_layers_), sizeof(num_layers_));

        for (std::size_t i = 0; i < num_layers_; ++i) {
            file.write(reinterpret_cast<const char*>(&layer_sizes_[i]), sizeof(layer_sizes_[i]));
        }

        for (std::size_t i = 0; i < num_layers_ - 1; ++i) {
            // Write weight matrix dimensions
            std::size_t rows = weights_[i].rows();
            std::size_t cols = weights_[i].cols();
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    double value = weights_[i](r, c);
                    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
                }
            }

            std::size_t bias_size = biases_[i].size();
            file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
            
            for (std::size_t j = 0; j < bias_size; ++j) {
                double value = biases_[i][j];
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
        
        return file.good();
    }
    
    bool NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
 
        std::size_t type_size;
        file.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
        
        std::string model_type(type_size, '\0');
        file.read(&model_type[0], type_size);
        
        if (model_type != get_model_type()) {
            return false;
        }

        file.read(reinterpret_cast<char*>(&num_layers_), sizeof(num_layers_));

        layer_sizes_.resize(num_layers_);
        for (std::size_t i = 0; i < num_layers_; ++i) {
            file.read(reinterpret_cast<char*>(&layer_sizes_[i]), sizeof(layer_sizes_[i]));
        }

        weights_.resize(num_layers_ - 1);
        biases_.resize(num_layers_ - 1);
        activations_.resize(num_layers_);
        z_values_.resize(num_layers_ - 1);

        for (std::size_t i = 0; i < num_layers_ - 1; ++i) {

            std::size_t rows, cols;
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

            weights_[i] = Matrix(rows, cols);

            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    double value;
                    file.read(reinterpret_cast<char*>(&value), sizeof(value));
                    weights_[i](r, c) = value;
                }
            }

            std::size_t bias_size;
            file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));

            biases_[i] = Vector(bias_size);
            
            for (std::size_t j = 0; j < bias_size; ++j) {
                double value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                biases_[i][j] = value;
            }
        }
        
        return file.good();
    }

    void NeuralNetwork::resetEarlyStopping() {
        early_stopping_counter_ = 0;
        best_validation_loss_ = std::numeric_limits<double>::max();
    }

    bool NeuralNetwork::shouldStopEarly(double validationLoss) {
        if (!early_stopping_enabled_) {
            return false;
        }
        
        if (validationLoss < best_validation_loss_ - early_stopping_min_delta_) {
            best_validation_loss_ = validationLoss;
            early_stopping_counter_ = 0;
            return false;
        } else {
            early_stopping_counter_++;
            return early_stopping_counter_ >= early_stopping_patience_;
        }
    }

    ActivationLayer::ActivationLayer(ActivationFunc activationFunc, 
                                   ActivationGradFunc gradFunc, 
                                   const std::string& name)
        : activation(activationFunc), activationGrad(gradFunc), name(name) {}

    Matrix ActivationLayer::forward(const Matrix& input) {
        lastInput = input;
        Matrix output(input.rows(), input.cols());
        
        for (std::size_t r = 0; r < input.rows(); ++r) {
            for (std::size_t c = 0; c < input.cols(); ++c) {
                output(r, c) = activation(input(r, c));
            }
        }
        
        return output;
    }

    Matrix ActivationLayer::backward(const Matrix& gradient, double learningRate) {
        Matrix gradInput(gradient.rows(), gradient.cols());
        
        for (std::size_t r = 0; r < gradient.rows(); ++r) {
            for (std::size_t c = 0; c < gradient.cols(); ++c) {
                gradInput(r, c) = gradient(r, c) * activationGrad(lastInput(r, c));
            }
        }
        
        return gradInput;
    }

    ConvolutionalLayer::ConvolutionalLayer(size_t inputChannels, size_t outputChannels, 
                                         size_t kernelSize, size_t stride, size_t padding,
                                         double initRange)
        : inputChannels(inputChannels), outputChannels(outputChannels), 
          kernelSize(kernelSize), stride(stride), padding(padding),
          biases(outputChannels, 0.0) {

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        // Scale for He initialization: sqrt(2 / (inputChannels * kernelSize * kernelSize))
        double scale = sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        
        // Create one filter per output channel
        for (size_t outC = 0; outC < outputChannels; ++outC) {
            // Each filter has dimensions [inputChannels, kernelSize, kernelSize]
            // But we store it flattened as a Matrix with shape [1, inputChannels * kernelSize * kernelSize]
            Matrix filter(1, inputChannels * kernelSize * kernelSize);
            
            // Initialize weights with He initialization
            for (size_t i = 0; i < inputChannels * kernelSize * kernelSize; ++i) {
                filter(0, i) = dist(rng) * scale;
            }
            
            filters.push_back(filter);
        }
    }

    void ConvolutionalLayer::reset() {
        lastInput = Matrix(0, 0);
        lastOutput = Matrix(0, 0);
    }

    Matrix ConvolutionalLayer::forward(const Matrix& input) {
        lastInput = input;

        Matrix output = applyConvolution(input);

        lastOutput = output;
        
        return output;
    }

    Matrix ConvolutionalLayer::applyConvolution(const Matrix& input) const {
        // Extract dimensions from input matrix
        // Assuming input is shaped as [batchSize, inputHeight * inputWidth * inputChannels]
        size_t batchSize = input.rows();

        size_t inputSize = std::sqrt(input.cols() / inputChannels);

        size_t outputSize = tensor_utils::get_conv_output_dim(inputSize, kernelSize, stride, padding);

        Matrix output(batchSize, outputSize * outputSize * outputChannels, 0.0);

        for (size_t batch = 0; batch < batchSize; ++batch) {
            // Use im2col to extract patches for efficient convolution
            Matrix patches = tensor_utils::im2col(
                input, batch, inputChannels, inputSize, inputSize, 
                kernelSize, stride, padding
            );

            for (size_t outC = 0; outC < outputChannels; ++outC) {
                for (size_t i = 0; i < patches.cols(); ++i) {
                    double sum = 0.0;
                    for (size_t j = 0; j < filters[outC].cols(); ++j) {
                        sum += filters[outC](0, j) * patches(j, i);
                    }

                    size_t outIndex = outC * outputSize * outputSize + i;
                    output(batch, outIndex) = sum + biases[outC];
                }
            }
        }
        
        return output;
    }

    Matrix ConvolutionalLayer::backward(const Matrix& gradient, double learningRate) {
        size_t batchSize = gradient.rows();
        size_t inputSize = std::sqrt(lastInput.cols() / inputChannels);
        size_t outputSize = std::sqrt(gradient.cols() / outputChannels);

        Matrix inputGradient(lastInput.rows(), lastInput.cols(), 0.0);

        std::vector<Matrix> filterGradients;
        for (size_t i = 0; i < outputChannels; ++i) {
            filterGradients.push_back(Matrix(1, filters[i].cols(), 0.0));
        }

        Vector biasGradients(outputChannels, 0.0);

        for (size_t batch = 0; batch < batchSize; ++batch) {
            Matrix patches = tensor_utils::im2col(
                lastInput, batch, inputChannels, inputSize, inputSize, 
                kernelSize, stride, padding
            );
            
            for (size_t outC = 0; outC < outputChannels; ++outC) {
                Matrix channelGradient(1, outputSize * outputSize);
                for (size_t i = 0; i < outputSize * outputSize; ++i) {
                    channelGradient(0, i) = gradient(batch, outC * outputSize * outputSize + i);

                    biasGradients[outC] += channelGradient(0, i);
                }
                
                for (size_t i = 0; i < filters[outC].cols(); ++i) {
                    for (size_t j = 0; j < channelGradient.cols(); ++j) {
                        filterGradients[outC](0, i) += patches(i, j) * channelGradient(0, j);
                    }
                }

                for (size_t i = 0; i < channelGradient.cols(); ++i) {
                    for (size_t j = 0; j < filters[outC].cols(); ++j) {
                        patches(j, i) = filters[outC](0, j) * channelGradient(0, i);
                    }
                }
            }

            Matrix batchGradient = tensor_utils::col2im(
                patches, 1, inputChannels, inputSize, inputSize, 
                kernelSize, stride, padding
            );

            for (size_t i = 0; i < inputGradient.cols(); ++i) {
                inputGradient(batch, i) += batchGradient(0, i);
            }
        }

        for (size_t outC = 0; outC < outputChannels; ++outC) {
            for (size_t i = 0; i < filters[outC].cols(); ++i) {
                filters[outC](0, i) -= learningRate * filterGradients[outC](0, i) / batchSize;
            }

            biases[outC] -= learningRate * biasGradients[outC] / batchSize;
        }
        
        return inputGradient;
    }

    MaxPoolingLayer::MaxPoolingLayer(size_t poolSize, size_t stride)
        : poolSize(poolSize), stride(stride == 0 ? poolSize : stride) {}

    Matrix MaxPoolingLayer::forward(const Matrix& input) {
        lastInput = input;

        size_t batchSize = input.rows();

        std::vector<size_t> shape = tensor_utils::infer_image_shape(input.cols());
        size_t channels = shape[0];
        size_t inputHeight = shape[1];
        size_t inputWidth = shape[2];

        size_t outputHeight = tensor_utils::get_pool_output_dim(inputHeight, poolSize, stride);
        size_t outputWidth = tensor_utils::get_pool_output_dim(inputWidth, poolSize, stride);

        Matrix output(batchSize, outputHeight * outputWidth * channels, 0.0);
        maxIndices = Matrix(batchSize, outputHeight * outputWidth * channels, 0.0);

        for (size_t batch = 0; batch < batchSize; ++batch) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t outY = 0; outY < outputHeight; ++outY) {
                    for (size_t outX = 0; outX < outputWidth; ++outX) {
                        size_t inStartY = outY * stride;
                        size_t inStartX = outX * stride;

                        double maxVal = -std::numeric_limits<double>::max();
                        size_t maxIdx = 0;
                        
                        for (size_t poolY = 0; poolY < poolSize; ++poolY) {
                            for (size_t poolX = 0; poolX < poolSize; ++poolX) {
                                size_t inY = inStartY + poolY;
                                size_t inX = inStartX + poolX;
                                
                                if (inY < inputHeight && inX < inputWidth) {
                                    size_t inIdx = tensor_utils::idx4d_to_flat(
                                        batch, c, inY, inX,
                                        channels, inputHeight, inputWidth
                                    );
                                    
                                    if (batch == 0 && inIdx >= input.cols()) {
                                        continue;
                                    }
                                    
                                    if (input(batch, c * inputHeight * inputWidth + inY * inputWidth + inX) > maxVal) {
                                        maxVal = input(batch, c * inputHeight * inputWidth + inY * inputWidth + inX);
                                        maxIdx = c * inputHeight * inputWidth + inY * inputWidth + inX;
                                    }
                                }
                            }
                        }

                        size_t outIdx = c * outputHeight * outputWidth + outY * outputWidth + outX;
                        output(batch, outIdx) = maxVal;
                        maxIndices(batch, outIdx) = maxIdx;
                    }
                }
            }
        }
        
        return output;
    }

    Matrix MaxPoolingLayer::backward(const Matrix& gradient, double learningRate) {

        size_t batchSize = gradient.rows();
        
        Matrix inputGradient(lastInput.rows(), lastInput.cols(), 0.0);

        for (size_t batch = 0; batch < batchSize; ++batch) {
            for (size_t i = 0; i < gradient.cols(); ++i) {
                size_t maxIdx = static_cast<size_t>(maxIndices(batch, i));
                inputGradient(batch, maxIdx) += gradient(batch, i);
            }
        }
        
        return inputGradient;
    }

    FlattenLayer::FlattenLayer() {}

    Matrix FlattenLayer::forward(const Matrix& input) {
        size_t batchSize = input.rows();
 
        std::vector<size_t> shape = tensor_utils::infer_image_shape(input.cols());
        inputShape = shape;

        return input;
    }

    Matrix FlattenLayer::backward(const Matrix& gradient, double learningRate) {
        return gradient;
    }

    DropoutLayer::DropoutLayer(double rate) : rate(rate) {}

    Matrix DropoutLayer::forward(const Matrix& input) {
        if (!training) {
            return input;
        }

        dropoutMask = Matrix(input.rows(), input.cols(), 0.0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        Matrix output(input.rows(), input.cols());
        
        for (size_t r = 0; r < input.rows(); ++r) {
            for (size_t c = 0; c < input.cols(); ++c) {
                if (dis(gen) > rate) {
                    dropoutMask(r, c) = 1.0;
                    output(r, c) = input(r, c) / (1.0 - rate);
                } else {
                    dropoutMask(r, c) = 0.0;
                    output(r, c) = 0.0;
                }
            }
        }
        
        return output;
    }

    Matrix DropoutLayer::backward(const Matrix& gradient, double learningRate) {
        Matrix gradInput(gradient.rows(), gradient.cols());

        for (size_t r = 0; r < gradient.rows(); ++r) {
            for (size_t c = 0; c < gradient.cols(); ++c) {
                gradInput(r, c) = gradient(r, c) * dropoutMask(r, c) / (1.0 - rate);
            }
        }
        
        return gradInput;
    }

    DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, double initRange)
        : inputSize(inputSize), outputSize(outputSize) {
        initializeWeights(initRange);
        weightMomentum = Matrix(outputSize, inputSize, 0.0);
        biasMomentum = Vector(outputSize, 0.0);
    }

    void DenseLayer::reset() {
        lastInput = Matrix(0, 0);
        lastOutput = Matrix(0, 0);
    }
    
    void DenseLayer::initializeWeights(double initRange) {
        weights = Matrix(outputSize, inputSize);
        biases = Vector(outputSize, 0.0);

        std::random_device rd;
        std::mt19937 gen(rd());
        double scale = std::sqrt(6.0 / (inputSize + outputSize));
        std::uniform_real_distribution<> dis(-scale, scale);
        
        for (size_t r = 0; r < weights.rows(); ++r) {
            for (size_t c = 0; c < weights.cols(); ++c) {
                weights(r, c) = dis(gen);
            }
        }
    }

    Matrix DenseLayer::forward(const Matrix& input) {
        lastInput = input;
        
        Matrix output(input.rows(), outputSize);
        
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < outputSize; ++j) {
                double sum = biases[j];
                for (size_t k = 0; k < inputSize; ++k) {
                    sum += weights(j, k) * input(i, k);
                }
                output(i, j) = sum;
            }
        }
        
        lastOutput = output;
        return output;
    }
    
    Matrix DenseLayer::backward(const Matrix& gradient, double learningRate) {
        Matrix weightGradients(weights.rows(), weights.cols(), 0.0);
        Vector biasGradients(biases.size(), 0.0);

        for (size_t n = 0; n < gradient.rows(); ++n) {
            for (size_t i = 0; i < outputSize; ++i) {
                double grad = gradient(n, i);

                biasGradients[i] += grad;

                for (size_t j = 0; j < inputSize; ++j) {
                    weightGradients(i, j) += grad * lastInput(n, j);
                }
            }
        }

        double batchSize = static_cast<double>(gradient.rows());
        
        for (size_t i = 0; i < outputSize; ++i) {
            biasMomentum[i] = momentumFactor * biasMomentum[i] + 
                             (1.0 - momentumFactor) * biasGradients[i] / batchSize;
            biases[i] -= learningRate * biasMomentum[i];

            for (size_t j = 0; j < inputSize; ++j) {
                weightMomentum(i, j) = momentumFactor * weightMomentum(i, j) + 
                                      (1.0 - momentumFactor) * weightGradients(i, j) / batchSize;
                weights(i, j) -= learningRate * weightMomentum(i, j);
            }
        }

        Matrix inputGradient(gradient.rows(), inputSize, 0.0);
        
        for (size_t n = 0; n < gradient.rows(); ++n) {
            for (size_t j = 0; j < inputSize; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < outputSize; ++i) {
                    sum += weights(i, j) * gradient(n, i);
                }
                inputGradient(n, j) = sum;
            }
        }
        
        return inputGradient;
    }

    BatchNormalizationLayer::BatchNormalizationLayer(size_t featureSize, double epsilon, double momentum)
        : featureSize(featureSize), epsilon(epsilon), momentum(momentum) {
        gamma = Vector(featureSize, 1.0);
        beta = Vector(featureSize, 0.0);

        runningMean = Vector(featureSize, 0.0);
        runningVar = Vector(featureSize, 1.0);
    }
    
    void BatchNormalizationLayer::reset() {
        lastInput = Matrix(0, 0);
        normalizedInput = Matrix(0, 0);
        batchMean = Vector(0);
        batchVar = Vector(0);
    }
    
    Matrix BatchNormalizationLayer::forward(const Matrix& input) {
        lastInput = input;
        size_t batchSize = input.rows();
        Matrix output(input.rows(), input.cols());
        
        if (training) {
            batchMean = Vector(featureSize, 0.0);
            
            for (size_t n = 0; n < batchSize; ++n) {
                for (size_t f = 0; f < featureSize; ++f) {
                    batchMean[f] += input(n, f) / batchSize;
                }
            }

            batchVar = Vector(featureSize, 0.0);
            for (size_t n = 0; n < batchSize; ++n) {
                for (size_t f = 0; f < featureSize; ++f) {
                    double diff = input(n, f) - batchMean[f];
                    batchVar[f] += (diff * diff) / batchSize;
                }
            }

            for (size_t f = 0; f < featureSize; ++f) {
                runningMean[f] = momentum * runningMean[f] + (1.0 - momentum) * batchMean[f];
                runningVar[f] = momentum * runningVar[f] + (1.0 - momentum) * batchVar[f];
            }

            normalizedInput = Matrix(input.rows(), input.cols());
            for (size_t n = 0; n < batchSize; ++n) {
                for (size_t f = 0; f < featureSize; ++f) {
                    double normalized = (input(n, f) - batchMean[f]) / std::sqrt(batchVar[f] + epsilon);
                    normalizedInput(n, f) = normalized;
                    output(n, f) = gamma[f] * normalized + beta[f];
                }
            }
        } else {
            for (size_t n = 0; n < batchSize; ++n) {
                for (size_t f = 0; f < featureSize; ++f) {
                    double normalized = (input(n, f) - runningMean[f]) / std::sqrt(runningVar[f] + epsilon);
                    output(n, f) = gamma[f] * normalized + beta[f];
                }
            }
        }
        
        return output;
    }
    
    Matrix BatchNormalizationLayer::backward(const Matrix& gradient, double learningRate) {
        size_t batchSize = gradient.rows();
        Matrix inputGradient(gradient.rows(), gradient.cols(), 0.0);

        Vector gammaGrad(featureSize, 0.0);
        Vector betaGrad(featureSize, 0.0);
        
        for (size_t n = 0; n < batchSize; ++n) {
            for (size_t f = 0; f < featureSize; ++f) {
                gammaGrad[f] += gradient(n, f) * normalizedInput(n, f);
                betaGrad[f] += gradient(n, f);
            }
        }

        for (size_t f = 0; f < featureSize; ++f) {
            gamma[f] -= learningRate * gammaGrad[f] / batchSize;
            beta[f] -= learningRate * betaGrad[f] / batchSize;
        }

        Matrix gradNormalized(gradient.rows(), gradient.cols());
        for (size_t n = 0; n < batchSize; ++n) {
            for (size_t f = 0; f < featureSize; ++f) {
                gradNormalized(n, f) = gradient(n, f) * gamma[f];
            }
        }

        Vector gradVar(featureSize, 0.0);
        for (size_t n = 0; n < batchSize; ++n) {
            for (size_t f = 0; f < featureSize; ++f) {
                double xMinusMean = lastInput(n, f) - batchMean[f];
                gradVar[f] += gradNormalized(n, f) * xMinusMean * 
                            -0.5 * std::pow(batchVar[f] + epsilon, -1.5);
            }
        }

        Vector gradMean = Vector(featureSize, 0.0);
        for (size_t n = 0; n < batchSize; ++n) {
            for (size_t f = 0; f < featureSize; ++f) {
                gradMean[f] += gradNormalized(n, f) * -1.0 / std::sqrt(batchVar[f] + epsilon);
            }
        }

        for (size_t f = 0; f < featureSize; ++f) {
            double sum = 0.0;
            for (size_t n = 0; n < batchSize; ++n) {
                sum += (lastInput(n, f) - batchMean[f]);
            }
            gradMean[f] += gradVar[f] * -2.0 * sum / batchSize;
        }

        for (size_t n = 0; n < batchSize; ++n) {
            for (size_t f = 0; f < featureSize; ++f) {
                inputGradient(n, f) = gradNormalized(n, f) / std::sqrt(batchVar[f] + epsilon)
                                   + gradVar[f] * 2.0 * (lastInput(n, f) - batchMean[f]) / batchSize
                                   + gradMean[f] / batchSize;
            }
        }
        
        return inputGradient;
    }
}