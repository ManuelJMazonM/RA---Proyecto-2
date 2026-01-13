#include "neuralnetwork.cpp"

using std::vector;

NeuralNetwork::NeuralNetwork(
    const vector<int>& topology,
    const ActivationType activation_type)
{
  layers.reserve(topology.length());
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
  for(auto l = layers.cbegin(); i != layers.cend(); i++)
  {
    output = l->forward(output, activation_type);
  }

  return output;
}
