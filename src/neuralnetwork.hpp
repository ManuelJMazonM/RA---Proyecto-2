#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "perceptron.hpp"
#include "layer.hpp"


class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& topology); // Ejemplo: {input, hidden, output}
    std::vector<double> predict(const std::vector<double>& inputs, ActivationType activation);
};

#endif
