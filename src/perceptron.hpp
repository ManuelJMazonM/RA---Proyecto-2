#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "common.hpp"


class Perceptron {
public:
    std::vector<double> weights;
    double bias;

    Perceptron(int input_size);
    double predict(const std::vector<double>& inputs, ActivationType activation);
};

#endif
