#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "neuralnetwork.hpp"
#include <vector>

using std::vector;
using std::cout;
using std::cerr;
using std::abs;


class Trainer {
public:
    static bool train(
        Perceptron& perceptron,
        const vector<vector<double>>& training_inputs,
        const vector<double>& training_outputs,
        double learning_rate = 0.1,
        int max_epochs = 1000,
        ActivationType activation = ActivationType::STEP);
    
    static double test_acc(
        NeuralNetwork& network,
        const vector<vector<double>>& test_inputs,
        const vector<double>& test_outputs);
};

#endif
