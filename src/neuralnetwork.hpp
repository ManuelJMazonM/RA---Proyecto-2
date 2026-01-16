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
  int predict(const vector<double>& inputs);
  int predicted_class() const;
  vector<double> raw_output() const;
};

#endif
