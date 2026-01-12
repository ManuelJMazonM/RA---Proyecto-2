#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "common.hpp"


class Perceptron {
public:
    std::vector<double> weights;
    double bias;

    Perceptron(int input_size);
    // Para Nivel 0 (PLA): Usa función escalón. Para Nivel 1: Usa activación derivable.
    double predict(const std::vector<double>& inputs, ActivationType activation);
};

#endif
