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


int NeuralNetwork::predict(const vector<double>& input)
{
  vector<double> output(input);
  for(auto l = layers.begin(); l != layers.end(); l++)
  {
    output = l->forward(output, activation_type);
  }

  return predicted_class();
}


int NeuralNetwork::predicted_class() const
{
  const vector<double>& raw = layers.back().last_outputs;
  if(raw.empty()) return -1;

  int max_index = 0;
  for(size_t i=1; i < raw.size(); i++)
  {
    if(raw[i] > raw[max_index]) max_index = i;
  }
  return max_index;
}


extern inline vector<double> NeuralNetwork::raw_output() const
{
  return vector<double>(layers.back().last_outputs);
}
