#pragma once
#include "nanoml/layer.hpp"
#include "nanoml/activations.hpp"
#include "nanoml/matrix.hpp"
#include "nanoml/model.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <string>
#include <cassert>

namespace ml {

using ActivationFunc = std::function<double(double)>;
using ActivationGradFunc = std::function<double(double)>;

class DenseLayer : public Layer {
    public:
        DenseLayer(size_t inputSize, size_t outputSize, double initRange = 0.1);
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return "Dense"; }
        void reset() override;

    private:
        Matrix weights;
        Vector biases;
        Matrix lastInput;
        Matrix lastOutput;
        size_t inputSize;
        size_t outputSize;

        Matrix weightMomentum;
        Vector biasMomentum;
        double momentumFactor = 0.9;

        void initializeWeights(double initRange);
};

class BatchNormalizationLayer : public Layer {
    public:
        BatchNormalizationLayer(size_t featureSize, double epsilon = 1e-8, double momentum = 0.9);
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return "BatchNorm"; }
        void reset() override;

        void setTrainingMode(bool isTraining) { training = isTraining; }

    private:
        size_t featureSize;
        double epsilon;
        double momentum;
        bool training = true;

        Vector gamma;
        Vector beta;
        
        Vector runningMean;
        Vector runningVar;

        Matrix lastInput;
        Matrix normalizedInput;
        Vector batchMean;
        Vector batchVar;
};

class ActivationLayer : public Layer {
    public:
        ActivationLayer(ActivationFunc activationFunc, ActivationGradFunc gradFunc, 
                    const std::string& name = "Activation");
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return name; }
        void reset() override { lastInput = Matrix(0, 0); }

    private:
        ActivationFunc activation;
        ActivationGradFunc activationGrad;
        Matrix lastInput;
        std::string name;
};

class ReLULayer : public ActivationLayer {
    public:
        ReLULayer() : ActivationLayer(relu, relu_grad, "ReLU") {}
    };

    class SigmoidLayer : public ActivationLayer {
    public:
        SigmoidLayer() : ActivationLayer(sigmoid, sigmoid_grad, "Sigmoid") {}
    };

    class TanhLayer : public ActivationLayer {
    public:
        TanhLayer() : ActivationLayer(tanh_activation, tanh_grad, "Tanh") {}
    };

    class DropoutLayer : public Layer {
    public:
        DropoutLayer(double rate);
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return "Dropout(" + std::to_string(rate) + ")"; }
        void reset() override { dropoutMask = Matrix(0, 0); }

    private:
        double rate;
        Matrix dropoutMask;
        bool training = true;

    public:
        void setTrainingMode(bool mode) { training = mode; }
        bool isTraining() const { return training; }
    };

    class ConvolutionalLayer : public Layer {
    public:
        ConvolutionalLayer(size_t inputChannels, size_t outputChannels, size_t kernelSize, 
                          size_t stride = 1, size_t padding = 0, double initRange = 0.1);
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return "Conv2D"; }
        void reset() override;
        
    private:
        size_t inputChannels;
        size_t outputChannels;
        size_t kernelSize;
        size_t stride;
        size_t padding;
        
        std::vector<Matrix> filters;
        Vector biases;

        Matrix lastInput;
        Matrix lastOutput;
        
        Matrix applyConvolution(const Matrix& input) const;
        Matrix computeConvolutionGradient(const Matrix& input, const Matrix& gradOutput);
    };
    
    class MaxPoolingLayer : public Layer {
    public:
        MaxPoolingLayer(size_t poolSize, size_t stride = 0);
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { 
            return "MaxPool(" + std::to_string(poolSize) + "x" + std::to_string(poolSize) + ")"; 
        }
        void reset() override { maxIndices = Matrix(0, 0); }
        
    private:
        size_t poolSize;
        size_t stride;
        Matrix maxIndices;
        Matrix lastInput;
    };

    class FlattenLayer : public Layer {
    public:
        FlattenLayer();
        
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradient, double learningRate) override;
        std::string getName() const override { return "Flatten"; }
        void reset() override { inputShape = {0, 0, 0}; }
        
    private:
        std::vector<size_t> inputShape;
    };

class NeuralNetwork : public Model {
    public:
        NeuralNetwork(const std::vector<std::size_t>& layer_sizes);
        
        Vector forward(const Matrix& X) const override;
        void backward(const Matrix& X, const Vector& dLdy, double lr) override;

        void setL1Regularization(double lambda) { l1_lambda_ = lambda; }
        void setL2Regularization(double lambda) { l2_lambda_ = lambda; }
        double getL1Regularization() const { return l1_lambda_; }
        double getL2Regularization() const { return l2_lambda_; }

        void enableEarlyStopping(size_t patience, double minDelta = 0.0) { 
            early_stopping_patience_ = patience;
            early_stopping_min_delta_ = minDelta; 
            early_stopping_enabled_ = true;
        }
        void disableEarlyStopping() { early_stopping_enabled_ = false; }
        bool isEarlyStoppingEnabled() const { return early_stopping_enabled_; }
        size_t getEarlyStoppingPatience() const { return early_stopping_patience_; }
        double getEarlyStoppingMinDelta() const { return early_stopping_min_delta_; }
        bool shouldStopEarly(double validationLoss);
        void resetEarlyStopping();

        bool save(const std::string& filename) const override;
        bool load(const std::string& filename) override;
        std::string get_model_type() const override { return "NeuralNetwork"; }
        
    private:
        std::vector<std::size_t> layer_sizes_;
        std::size_t num_layers_;
        mutable std::vector<Matrix> weights_;
        mutable std::vector<Vector> biases_;
        mutable std::vector<Vector> activations_;
        mutable std::vector<Vector> z_values_;
        mutable std::mt19937 rng_;

        double l1_lambda_ = 0.0;
        double l2_lambda_ = 0.0;

        bool early_stopping_enabled_ = false;
        size_t early_stopping_patience_ = 10;
        double early_stopping_min_delta_ = 0.0;
        size_t early_stopping_counter_ = 0;
        double best_validation_loss_ = std::numeric_limits<double>::max();
    };
}