#include "trainer.hpp"
#include "common.hpp"
#include <iostream>

using std::cout;
using std::cerr;
using std::abs;
using std::exp;
using std::max;


// This function is to distinguish what the output should be with 1 out neuron or more
static double targetToDouble(int target_class, ActivationType activation_type, size_t output_size, size_t neuron_index = 0)
{
  if (output_size == 1) {
    if (activation_type == ActivationType::STEP) {
      return (target_class > 0) ? 1.0 : -1.0;
    } else if (activation_type == ActivationType::SIGMOID) {
      return (target_class > 0) ? 1.0 : 0.0;
    } else {
      return static_cast<double>(target_class);
    }
  } else {
    return (neuron_index == static_cast<size_t>(target_class)) ? 1.0 : 0.0;
  }
}

vector<vector<double>> Trainer::computeDeltas(NeuralNetwork& nn, int target_class) const
{
  int num_layers = nn.layers.size();
  vector<vector<double>> deltas(num_layers); 

  int L = num_layers - 1;
  int num_neurons_L = nn.layers[L].neurons.size(); 
  deltas[L].resize(num_neurons_L);

  if (nn.activation_type == ActivationType::STEP) {
    cerr << "Warning: STEP activation has zero derivative. No backpropagation\n";
    return deltas;
  }

  for (int j = 0; j < num_neurons_L; ++j)
  {
    double x_j_L = nn.layers[L].last_outputs[j];
    double target = targetToDouble(target_class, nn.activation_type, num_neurons_L, j);
    double error_derivative = 2.0 * (x_j_L - target);
    double theta_prime = Activations::applyDerivative(x_j_L, nn.activation_type);

    deltas[L][j] = theta_prime * error_derivative;
  }

  for (int l = L - 1; l >= 1; l--)  // Empezar desde 1, no desde 0 (la capa 0 es entrada)
  {
    int num_neurons_l = nn.layers[l].neurons.size();
    deltas[l].resize(num_neurons_l);

    for (int i = 0; i < num_neurons_l; ++i) {
      double sum_deltas_next = 0.0;

      for (size_t j = 0; j < nn.layers[l+1].neurons.size(); ++j){
        sum_deltas_next += nn.layers[l+1].neurons[j].weights[i] * deltas[l+1][j];
      }

      double x_i_l = nn.layers[l].last_outputs[i];
      double theta_prime = Activations::applyDerivative(x_i_l, nn.activation_type);

      deltas[l][i] = theta_prime * sum_deltas_next;
    }
  }

  return deltas; 
}


void Trainer::applyGradients(NeuralNetwork& nn, const std::vector<std::vector<double>>& deltas, double eta) const
{
  // Empezar desde capa 1 (la capa 0 es la capa de entrada, sin pesos entrenables)
  for (size_t l = 1; l < nn.layers.size(); ++l){
    Layer& layer = nn.layers[l];

    for (size_t j = 0; j < layer.neurons.size(); ++j){
      Perceptron& neuron = layer.neurons[j];
      double delta_j = deltas[l][j];

      for (size_t k = 0; k < neuron.weights.size(); ++k){
        double x_k = layer.last_inputs[k];
        neuron.weights[k] -= eta * delta_j * x_k;
      }

      neuron.bias -= eta * delta_j;
    }
  }
}


double Trainer::train(
    NeuralNetwork& nn,
    const Dataset& ds,
    const int epochs) const
{
  if (ds.train_inputs.empty() || ds.train_inputs.size() != ds.train_targets.size()) {
    cerr << "Error: Invalid training data\n";
    return 0.0;
  }

  for (int e = 0; e < epochs; ++e)
  {
    for (size_t i = 0; i < ds.train_inputs.size(); ++i)
    {
      nn.predict(ds.train_inputs[i]);
      vector<vector<double>> deltas = computeDeltas(nn, ds.train_targets[i]);
      applyGradients(nn, deltas, learning_rate); 
    }

    if (!ds.val_inputs.empty() && (e%10 == 0 || e == epochs-1)){
      double v_acc = test_acc(nn, ds.val_inputs, ds.val_targets);
      cout << "Epoca " << e << "| Prec. Validacion: " << v_acc << "%" << std::endl;
    }  
  }

  return test_acc(nn, ds.train_inputs, ds.train_targets); 
}


double Trainer::train(
    Perceptron& perceptron,
    const vector<vector<double>>& training_inputs,
    const vector<int>& training_outputs,
    double learning_rate,
    int max_epochs,
    ActivationType activation) const
{
  if (training_inputs.empty() || training_inputs.size() != training_outputs.size()) {
    cerr << "Error: Invalid training data\n";
    return 0.0;
  }

  size_t input_size = training_inputs[0].size();
  if (perceptron.weights.size() != input_size) {
    cerr << "Error: Perceptron input size doesn't match training data\n";
    return 0.0;
  }

  for (int epoch = 0; epoch < max_epochs; ++epoch) {
    for (size_t i = 0; i < training_inputs.size(); ++i) {
      double prediction = perceptron.predict(training_inputs[i], activation);
      double target;
      if (activation == ActivationType::STEP) {
        target = (training_outputs[i] > 0) ? 1.0 : -1.0;
      } else if (activation == ActivationType::SIGMOID) {
        target = (training_outputs[i] > 0) ? 1.0 : 0.0;
      } else {
        target = static_cast<double>(training_outputs[i]);
      }
      
      double error = target - prediction;
      
      if (abs(error) > 0.01) {
        for (size_t j = 0; j < input_size; ++j) {
          perceptron.weights[j] += learning_rate * error * training_inputs[i][j];
        }
        perceptron.bias += learning_rate * error;
      }
    }
  }

  int hits = 0;
  for (size_t i = 0; i < training_inputs.size(); ++i) {
    double prediction = perceptron.predict(training_inputs[i], activation);
    int predicted_class;
    
    if (activation == ActivationType::STEP) {
      predicted_class = (prediction >= 0.0) ? 1 : -1;
    } else if (activation == ActivationType::SIGMOID) {
      predicted_class = (prediction >= 0.5) ? 1 : 0;
    } else {
      predicted_class = static_cast<int>(round(prediction));
    }
    
    if (predicted_class == training_outputs[i]) {
      hits++;
    }
  }

  double accuracy = 100.0 * hits / training_inputs.size();
  return accuracy;
}


double Trainer::test_acc(
    NeuralNetwork& network,
    const vector<vector<double>>& test_inputs,
    const vector<int>& test_outputs) const  // Changed to int
{
  if (network.layers.empty() || test_inputs.size() != test_outputs.size()) {
    return 0.0;
  }

  int hits = 0;
  for (size_t i = 0; i < test_inputs.size(); ++i) {
    if (network.predict(test_inputs[i]) == test_outputs[i]) {
      hits++;
    }
  }

  return 100.0 * hits / test_inputs.size();
}
