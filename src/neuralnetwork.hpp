#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "perceptron.hpp"
#include "layer.hpp"

using std::vector;

class NeuralNetwork {
public:
  vector<Layer> layers;
  ActivationType activation_type;

  NeuralNetwork(const vector<int>& topology, const ActivationType activation_type);
  vector<double> predict(const vector<double>& inputs);
};

#endif
