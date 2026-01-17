#include "neuralnetwork.hpp"


NeuralNetwork::NeuralNetwork(
    const vector<int>& topology,
    const ActivationType activation_type)
{
  layers.reserve(topology.size());
  int ipn = 0;
  for(auto i = topology.cbegin(); i != topology.cend(); i++)
  {
    layers.emplace_back(*i, ipn);
    ipn = *i;
  }

  this->activation_type = activation_type;
}


vector<double> NeuralNetwork::predict(const vector<double>& input)
{
  vector<double> output(input);
  for(auto l = layers.begin(); l != layers.end(); l++)
  {
    output = l->forward(output, activation_type);
  }

  return output;
}
