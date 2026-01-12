#ifndef LAYER_HPP
#define LAYER_HPP

#include "perceptron.hpp"


class Layer {
public:
    std::vector<Perceptron> neurons;
    std::vector<double> last_outputs; //TODO: Guardar para Backpropagation

    Layer(int num_neurons, int inputs_per_neuron);
    std::vector<double> forward(const std::vector<double>& inputs, ActivationType activation);
};

#endif
