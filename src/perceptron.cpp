#include "perceptron.hpp"
#include <random>

Perceptron::Perceptron(int input_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    
    weights.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
        weights[i] = dist(gen);
    }
    bias = dist(gen);
}

double Perceptron::predict(const std::vector<double>& inputs, ActivationType activation) {
    double s = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        s += weights[i] * inputs[i];
    }
    return Activations::apply(s, activation);
}
